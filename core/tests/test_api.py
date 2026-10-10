from datetime import date, timedelta

import pytest
from conftest import FakeLLM, FakeRecorder, make_wav
from fastapi.testclient import TestClient

from earmark.api.app import create_app
from earmark.llm.base import LLMError

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(ctx):
    with TestClient(create_app(ctx), headers=AUTH) as client:
        yield client


def test_health_is_open_but_everything_else_needs_the_token(ctx):
    with TestClient(create_app(ctx)) as anonymous:
        assert anonymous.get("/health").json()["status"] == "ok"
        assert anonymous.get("/status").status_code == 401
        assert anonymous.post("/recording/start").status_code == 401
        assert anonymous.get("/tasks", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert not ctx.recorder.running


def test_record_stop_and_process_end_to_end(client, ctx, transcriber, llm):
    started = client.post("/recording/start", json={"title": "Calculus I"})
    assert started.status_code == 201
    session_id = started.json()["session"]["id"]
    assert started.json()["mic"] is True and started.json()["system"] is True

    status = client.get("/status").json()
    assert status["recording"] is True and status["session_id"] == session_id
    assert status["levels"] == {"mic": -20.0, "system": -30.0}
    assert client.post("/recording/start").status_code == 409

    stopped = client.post("/recording/stop", params={"wait": True})
    assert stopped.status_code == 200
    body = stopped.json()
    assert body["session"]["status"] == "done"
    assert body["session"]["title"] == "Calculus I"  # the user's title wins over the model's
    assert body["session"]["duration_s"] == 1.0
    assert [task["title"] for task in body["tasks"]] == ["Finish problem set 3", "Review limits"]
    assert sorted(transcriber.calls) == ["mic-00000.wav", "system-00000.wav"]
    assert client.get("/status").json()["recording"] is False

    transcript = client.get(f"/sessions/{session_id}/transcript").json()
    assert {row["source"] for row in transcript} == {"mic", "system"}
    assert [s["id"] for s in client.get("/sessions").json()] == [session_id]
    assert client.post("/recording/stop").status_code == 409


def test_stop_can_skip_processing_and_process_later(client, llm):
    session_id = client.post("/recording/start").json()["session"]["id"]

    stopped = client.post("/recording/stop", params={"process": False}).json()
    assert stopped["session"]["status"] == "recorded" and llm.prompts == []

    processed = client.post(f"/sessions/{session_id}/process", params={"wait": True}).json()
    assert processed["session"]["status"] == "done"
    assert processed["session"]["title"] == "Derivatives"


def test_summarize_again_with_overrides_reuses_the_transcript(client, ctx, transcriber):
    session_id = client.post("/recording/start").json()["session"]["id"]
    client.post("/recording/stop", params={"wait": True})
    seen = {}

    def factory(settings, _env):
        seen["provider"], seen["model"] = settings.llm_provider, settings.llm_model
        return FakeLLM()

    ctx.llm_factory = factory
    calls_before = list(transcriber.calls)

    response = client.post(
        f"/sessions/{session_id}/summarize",
        params={"wait": True},
        json={"llm_provider": "anthropic", "llm_model": "claude-test"},
    )

    assert response.json()["session"]["status"] == "done"
    assert seen == {"provider": "anthropic", "model": "claude-test"}
    assert transcriber.calls == calls_before
    # A one-off override does not change the saved settings.
    assert client.get("/settings").json()["llm_provider"] == "omlx"


def test_failed_processing_is_reported_on_the_session(client, ctx):
    def offline(_settings, _env):
        raise LLMError("could not connect to omlx at http://localhost:8000")

    ctx.llm_factory = offline
    client.post("/recording/start")

    body = client.post("/recording/stop", params={"wait": True}).json()

    assert body["session"]["status"] == "failed"
    assert "could not connect to omlx" in body["session"]["error"]


def test_start_without_the_helper_is_a_conflict_and_leaves_nothing_behind(client, ctx):
    ctx.recorder = FakeRecorder(available=False)
    ctx.recorder.start = FakeRecorder.__mro__[1].start.__get__(ctx.recorder)  # the real start()

    response = client.post("/recording/start")

    assert response.status_code == 409
    assert "capture helper not found" in response.json()["detail"]
    assert client.get("/sessions").json() == []


def test_import_copies_the_file_and_processes_it(client, tmp_path, transcriber):
    source = make_wav(tmp_path / "recorded lecture.wav", 1.0)

    response = client.post("/sessions/import", params={"wait": True}, json={"path": str(source)})

    assert response.status_code == 201
    assert response.json()["session"]["status"] == "done"
    assert response.json()["session"]["title"] == "recorded lecture"
    assert transcriber.calls == ["system-00000.wav"]
    assert source.exists()
    assert client.post("/sessions/import", json={"path": str(tmp_path / "nope.wav")}).status_code == 404


def test_settings_roundtrip_and_validation(client):
    assert client.get("/settings").json()["llm_provider"] == "omlx"

    updated = client.put(
        "/settings", json={"llm_provider": "anthropic", "language": "en", "chunk_chars": 6000}
    )
    assert updated.status_code == 200
    assert client.get("/settings").json() | {} == updated.json()
    assert updated.json()["llm_provider"] == "anthropic" and updated.json()["language"] == "en"
    assert updated.json()["stt_provider"] == "mlx-whisper"  # untouched

    assert client.put("/settings", json={"language": None}).json()["language"] is None
    assert client.put("/settings", json={"llm_provider": "skynet"}).status_code == 422
    assert client.put("/settings", json={"segment_seconds": 1}).status_code == 422


def test_tasks_crud_and_agenda(client):
    today = date.today()

    def add(title, due=None, **extra):
        payload = {"title": title, "due_date": due.isoformat() if due else None, **extra}
        response = client.post("/tasks", json=payload)
        assert response.status_code == 201
        return response.json()["id"]

    overdue = add("Hand in the report", today - timedelta(days=1))
    add("Read chapter 4", today)
    add("Problem set", today + timedelta(days=3), priority="high")
    add("Final exam", today + timedelta(days=30))
    add("Organize notes")
    assert client.post("/tasks", json={"title": ""}).status_code == 422
    assert client.post("/tasks", json={"title": "x", "session_id": 999}).status_code == 404

    agenda = client.get("/agenda", params={"days": 7}).json()
    assert agenda["day"] == today.isoformat()
    assert {
        key: [t["title"] for t in agenda[key]]
        for key in ("overdue", "today", "upcoming", "later", "unscheduled")
    } == {
        "overdue": ["Hand in the report"],
        "today": ["Read chapter 4"],
        "upcoming": ["Problem set"],
        "later": ["Final exam"],
        "unscheduled": ["Organize notes"],
    }

    done = client.patch(f"/tasks/{overdue}", json={"status": "done"}).json()
    assert done["status"] == "done" and done["completed_at"] is not None
    assert "Hand in the report" not in [t["title"] for t in client.get("/tasks").json()]
    assert len(client.get("/tasks", params={"all": True}).json()) == 5
    assert [t["title"] for t in client.get("/tasks", params={"status": "done"}).json()] == [
        "Hand in the report"
    ]

    reopened = client.patch(
        f"/tasks/{overdue}", json={"status": "open", "due_date": None, "title": None}
    ).json()
    assert reopened["status"] == "open" and reopened["completed_at"] is None
    assert reopened["due_date"] is None and reopened["title"] == "Hand in the report"

    assert client.delete(f"/tasks/{overdue}").status_code == 204
    assert client.patch(f"/tasks/{overdue}", json={"status": "done"}).status_code == 404


def test_delete_session_removes_its_data_and_audio(client, ctx):
    session_id = client.post("/recording/start").json()["session"]["id"]
    assert client.delete(f"/sessions/{session_id}").status_code == 409  # still recording
    client.post("/recording/stop", params={"wait": True})
    audio_dir = ctx.env.sessions_dir / f"{session_id:06d}"
    assert audio_dir.is_dir()

    assert client.delete(f"/sessions/{session_id}").status_code == 204

    assert not audio_dir.exists()
    assert client.get(f"/sessions/{session_id}").status_code == 404
    assert client.get("/tasks", params={"all": True}).json() == []


def test_providers_report(client, ctx):
    report = client.get("/providers").json()

    assert report["capture"]["available"] is True
    assert report["stt"] == {"provider": "mlx-whisper", "model": "fake-whisper", "available": True}
    assert report["llm"]["available"] is True and report["llm"]["model"] == "fake-model"


def test_shutdown_stops_an_active_recording(ctx):
    with TestClient(create_app(ctx), headers=AUTH) as client:
        client.post("/recording/start")
        assert ctx.recorder.running
    assert not ctx.recorder.running


def test_summarize_without_a_transcript_is_refused_up_front(client):
    client.post("/recording/start")
    session_id = client.post("/recording/stop", params={"process": False}).json()["session"]["id"]

    response = client.post(f"/sessions/{session_id}/summarize")

    assert response.status_code == 409
    assert "no transcript yet" in response.json()["detail"]
    assert client.get(f"/sessions/{session_id}").json()["session"]["status"] == "recorded"


def test_manual_tasks_are_marked_as_such_and_reminders_sync_over_http(client, exporter):
    created = client.post("/tasks", json={"title": "Buy a notebook"}).json()
    assert created["origin"] == "manual" and created["reminder_id"] is None

    client.post("/recording/start")
    body = client.post("/recording/stop", params={"wait": True}).json()
    assert body["session"]["note_name"] and body["session"]["export_error"] is None
    assert all(task["origin"] == "extracted" and task["reminder_id"] for task in body["tasks"])

    next(iter(exporter.reminders.values()))["done"] = True
    assert client.post("/reminders/sync").json() == {"completed": 1, "dismissed": 0}

    exporter.fail = "Reminders: not allowed"
    assert client.post("/reminders/sync").status_code == 502
