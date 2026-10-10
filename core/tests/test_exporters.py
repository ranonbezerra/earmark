import asyncio
import json
import subprocess
from datetime import UTC, date, datetime

import pytest
from conftest import FakeLLM, make_wav
from sqlmodel import Session, select

from earmark import service
from earmark.config import UserSettings
from earmark.exporters import ExportError, apple, build_exporter, export_session, render_section
from earmark.models import SessionStatus, StudySession, Task, TaskOrigin, TaskStatus

NOON = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)


def processed_session(ctx, tmp_path, **settings):
    """Imports a file and runs the whole pipeline with the fakes in ``ctx``."""
    if settings:
        ctx.settings_store.update(settings)
    session = service.import_audio(ctx, make_wav(tmp_path / "lecture.wav", 1.0), title="Calculus")
    return asyncio.run(service.run_processing(ctx, session.id))


def tasks_of(ctx, session_id):
    with Session(ctx.engine) as db:
        return list(db.exec(select(Task).where(Task.session_id == session_id).order_by(Task.id)))


def test_section_is_plain_html_with_escaped_text():
    session = StudySession(
        id=1,
        title="Limits & <derivatives>",
        started_at=NOON,
        summary="## Chain rule\n\nUse it **whenever** functions are composed.\n\n- item one\n- item two",
        key_points=["a < b"],
        open_questions=[],
    )
    tasks = [Task(title="Do problem set 3", due_date=date(2026, 10, 12), origin=TaskOrigin.EXTRACTED)]

    html = render_section(session, tasks, UserSettings())

    assert "· Limits &amp; &lt;derivatives&gt;</h2>" in html
    assert "<h3>Chain rule</h3>" in html and "<h2>Chain" not in html
    assert "<strong>whenever</strong>" in html and "<li>item one</li>" in html
    assert "<h3>Key points</h3><ul><li>a &lt; b</li></ul>" in html
    assert "Open questions" not in html
    assert "<h3>Tasks</h3><ul><li>Do problem set 3 (due 2026-10-12)</li></ul>" in html


def test_processing_writes_the_day_note_and_one_reminder_per_task(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)

    assert session.status == SessionStatus.DONE and session.export_error is None
    day = session.started_at.astimezone().strftime("%Y-%m-%d")
    assert session.note_name == f"Earmark {day}"
    (note,) = exporter.notes.values()
    assert (note["folder"], note["name"]) == ("Earmark", f"Earmark {day}")
    assert "Calculus</h2>" in note["body"] and "The derivative measures the rate of change." in note["body"]

    reminders = list(exporter.reminders.values())
    assert [(r["list"], r["title"], r["due"], r["high"]) for r in reminders] == [
        ("Earmark", "Finish problem set 3", date(2026, 10, 12), True),
        ("Earmark", "Review limits", None, False),
    ]
    assert reminders[0]["notes"].startswith("Exercises 1 to 10\n\nSession: Calculus (")
    assert reminders[0]["notes"].endswith(f"Note: Earmark / Earmark {day}")
    assert [t.reminder_id for t in tasks_of(ctx, session.id)] == list(exporter.reminders)


def test_two_sessions_on_the_same_day_share_one_note(ctx, exporter, tmp_path):
    processed_session(ctx, tmp_path)
    processed_session(ctx, tmp_path)

    (note,) = exporter.notes.values()
    assert note["body"].count("Calculus</h2>") == 2


def test_session_grouping_writes_one_note_per_session_and_replaces_it(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path, notes_grouping="session")
    asyncio.run(service.run_processing(ctx, session.id, skip_transcription=True))
    processed_session(ctx, tmp_path)

    assert len(exporter.notes) == 2
    first = exporter.notes[session.note_id]
    assert first["name"].startswith("Earmark 20") and first["name"].endswith(" · Calculus")
    assert first["body"].count("The derivative measures") == 1  # replaced, not appended


def test_summarizing_again_swaps_reminders_without_duplicating(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)
    done, _other = tasks_of(ctx, session.id)
    service.update_task(ctx, done.id, {"status": TaskStatus.DONE})
    assert exporter.reminders[done.reminder_id]["done"] is True

    ctx.llm_factory = lambda _s, _e: FakeLLM(
        json.dumps({"title": "x", "summary": "new", "tasks": [{"title": "Watch lecture 5"}]})
    )
    asyncio.run(service.run_processing(ctx, session.id, skip_transcription=True))

    assert sorted(r["title"] for r in exporter.reminders.values()) == [
        "Finish problem set 3",
        "Watch lecture 5",
    ]
    (note,) = exporter.notes.values()
    assert "(updated)" in note["body"]


def test_export_failure_keeps_the_session_done_and_says_why(ctx, exporter, tmp_path):
    exporter.fail = "Notes: not allowed to control Notes"

    session = processed_session(ctx, tmp_path)

    assert session.status == SessionStatus.DONE
    assert session.export_error == "Notes: not allowed to control Notes"
    assert session.summary and len(tasks_of(ctx, session.id)) == 2


def test_export_can_be_turned_off(ctx, exporter, tmp_path):
    processed_session(ctx, tmp_path, export_notes=False, export_reminders=True)
    assert exporter.notes == {} and len(exporter.reminders) == 2
    assert "Note:" not in next(iter(exporter.reminders.values()))["notes"]

    assert build_exporter(UserSettings(export_notes=False, export_reminders=False)) is None


def test_export_is_idempotent_for_reminders(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)

    export_session(session.id, engine=ctx.engine, settings=ctx.settings_store.load(), exporter=exporter)

    assert len(exporter.reminders) == 2


def test_sync_brings_completions_and_deletions_back(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)
    first, second = tasks_of(ctx, session.id)
    exporter.reminders[first.reminder_id]["done"] = True
    del exporter.reminders[second.reminder_id]

    assert service.sync_reminders(ctx) == {"completed": 1, "dismissed": 1}

    first, second = tasks_of(ctx, session.id)
    assert first.status == TaskStatus.DONE and first.completed_at is not None
    assert second.status == TaskStatus.DISMISSED
    assert service.sync_reminders(ctx) == {"completed": 0, "dismissed": 0}


def test_sync_changes_nothing_when_reminders_cannot_be_read(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)
    exporter.fail = "Reminders: not allowed"

    with pytest.raises(ExportError):
        service.sync_reminders(ctx)

    assert all(t.status == TaskStatus.OPEN for t in tasks_of(ctx, session.id))


def test_deleting_a_task_or_a_session_removes_its_reminders(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)
    first, _second = tasks_of(ctx, session.id)

    service.delete_task(ctx, first.id)
    assert len(exporter.reminders) == 1

    service.delete_session(ctx, session.id)
    assert exporter.reminders == {}


def test_reminders_failure_never_breaks_task_updates(ctx, exporter, tmp_path):
    session = processed_session(ctx, tmp_path)
    first, _second = tasks_of(ctx, session.id)
    exporter.fail = "Reminders: not allowed"

    updated = service.update_task(ctx, first.id, {"status": TaskStatus.DONE})

    assert updated.status == TaskStatus.DONE


def test_apple_exporter_passes_text_as_arguments_never_inside_the_script(monkeypatch):
    calls = []

    def fake_run(command, *, input, capture_output, text, timeout):  # noqa: A002
        calls.append((command, input))
        return subprocess.CompletedProcess(command, 0, stdout="x-apple-reminder://ABC\n", stderr="")

    monkeypatch.setattr(apple.subprocess, "run", fake_run)
    hostile = 'Read "ch. 4"\' & do shell script "rm -rf ~"'

    reminder_id = apple.AppleExporter().create_reminder(
        list_name="earmark", title=hostile, notes="n", due=date(2026, 10, 12), high_priority=True
    )

    (command, script) = calls[0]
    assert reminder_id == "x-apple-reminder://ABC"
    assert command == [apple.OSASCRIPT, "-", "earmark", hostile, "n", "2026", "10", "12", "1"]
    assert "rm -rf" not in script and hostile not in script


def test_apple_exporter_explains_missing_permission_and_timeouts(monkeypatch):
    def denied(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="execution error: Not authorized to send Apple events to Notes. (-1743)\n",
        )

    monkeypatch.setattr(apple.subprocess, "run", denied)
    with pytest.raises(ExportError, match="Notes: not allowed to control Notes"):
        apple.AppleExporter().append_to_note(folder="f", name="n", section_html="<p>x</p>")

    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(apple.subprocess, "run", hang)
    with pytest.raises(ExportError, match="Reminders did not answer"):
        apple.AppleExporter(timeout=1).reminder_states(["a"])


def test_apple_exporter_parses_reminder_states(monkeypatch):
    def states(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="a\tdone\nb\topen\nc\tmissing\n", stderr="")

    monkeypatch.setattr(apple.subprocess, "run", states)

    assert apple.AppleExporter().reminder_states(["a", "b", "c"]) == {
        "a": "done",
        "b": "open",
        "c": "missing",
    }
    assert apple.AppleExporter().reminder_states([]) == {}


def test_note_title_is_the_app_prefix_plus_the_date_and_both_are_configurable():
    from earmark.exporters import note_name

    session = StudySession(id=1, title="Calculus", started_at=NOON)
    day = NOON.astimezone()

    assert note_name(session, UserSettings()) == f"Earmark {day:%Y-%m-%d}"
    custom = UserSettings(note_title_prefix="Study log", note_date_format="%b %d, %Y")
    assert note_name(session, custom) == f"Study log {day:%b %d, %Y}"
    per_session = UserSettings(notes_grouping="session")
    assert note_name(session, per_session) == f"Earmark {day:%Y-%m-%d} · Calculus"
