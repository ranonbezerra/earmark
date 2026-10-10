import asyncio
import json
from datetime import date
from pathlib import Path

from conftest import FakeLLM, FakeTranscriber, make_wav
from sqlmodel import Session, select

from ouvinte.llm.base import LLMError
from ouvinte.models import SessionStatus, StudySession, Task, TaskStatus, TranscriptSegment
from ouvinte.pipeline import TimedText, drop_mic_echo, format_timestamp, process_session
from ouvinte.transcription.base import TranscriptChunk

TODAY = date(2026, 10, 10)


def new_session(engine, audio_dir: Path) -> int:
    with Session(engine) as db:
        session = StudySession(status=SessionStatus.RECORDED, audio_dir=str(audio_dir))
        db.add(session)
        db.commit()
        return session.id


def run(session_id, *, engine, env, settings, transcriber, llm, exporter=None, **kwargs):
    kwargs.setdefault("today", TODAY)
    return asyncio.run(
        process_session(
            session_id,
            engine=engine,
            env=env,
            settings=settings,
            transcriber_factory=lambda _s, _e: transcriber,
            llm_factory=lambda _s, _e: llm,
            exporter_factory=lambda _s: exporter,
            **kwargs,
        )
    )


def rows(engine, model, session_id):
    with Session(engine) as db:
        return list(db.exec(select(model).where(model.session_id == session_id)))


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3725.9) == "01:02:05"


def test_mic_echo_of_the_speakers_is_dropped_but_own_speech_is_kept():
    chunks = [
        TimedText("system", 10.0, 16.0, "A derivada de uma função mede a taxa de variação instantânea."),
        TimedText("mic", 10.5, 16.5, "a derivada de uma função mede a taxa de variação instantânea"),
        TimedText("mic", 17.0, 20.0, "Preciso revisar isso antes da prova de sexta."),
        TimedText("mic", 11.0, 12.0, "taxa de variação"),  # too short to judge
        TimedText("mic", 300.0, 306.0, "a derivada de uma função mede a taxa de variação instantânea"),
    ]

    kept = drop_mic_echo(chunks)

    assert [(c.source, c.start_s) for c in kept] == [
        ("system", 10.0),
        ("mic", 17.0),
        ("mic", 11.0),
        ("mic", 300.0),
    ]


def test_full_pipeline_stores_transcript_summary_and_tasks(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    make_wav(audio / "system-00001.wav", 1.0, kind="silence")
    make_wav(audio / "mic-00000.wav", 1.0)
    (audio / "segments.jsonl").write_text(
        "".join(
            json.dumps({"track": t, "index": i, "file": f"{t}-{i:05d}.wav", "start_offset": o}) + "\n"
            for t, i, o in (("system", 0, 0.0), ("mic", 0, 0.2), ("system", 1, 300.0))
        )
    )
    transcriber = FakeTranscriber(
        {
            "system-00000.wav": [TranscriptChunk(0.1, 0.9, "A derivada mede a taxa de variação.")],
            "mic-00000.wav": [TranscriptChunk(0.3, 0.8, "Tenho que fazer a lista três até segunda.")],
        }
    )
    llm = FakeLLM()
    session_id = new_session(engine, audio)

    session = run(session_id, engine=engine, env=env, settings=settings, transcriber=transcriber, llm=llm)

    assert session.status == SessionStatus.DONE and session.error is None
    assert session.title == "Derivadas"
    assert session.summary.startswith("## Derivadas")
    assert session.key_points == ["Regra da cadeia"]
    assert (session.stt_provider, session.llm_provider, session.llm_model) == (
        "fake-stt",
        "fake-llm",
        "fake-model",
    )
    assert session.duration_s == 301.0
    # The silent segment never reaches the model.
    assert transcriber.calls == ["system-00000.wav", "mic-00000.wav"]
    transcript = sorted(rows(engine, TranscriptSegment, session_id), key=lambda r: r.start_s)
    assert [(r.source, round(r.start_s, 2)) for r in transcript] == [("system", 0.1), ("mic", 0.5)]
    assert "[00:00:00] [mic] Tenho que fazer a lista três até segunda." in llm.prompts[0]
    tasks = sorted(rows(engine, Task, session_id), key=lambda t: t.id)
    assert [(t.title, t.due_date, t.priority.value, t.status) for t in tasks] == [
        ("Resolver a lista 3", date(2026, 10, 12), "high", TaskStatus.OPEN),
        ("Revisar limites", None, "medium", TaskStatus.OPEN),
    ]
    assert transcriber.closed and llm.closed


def test_summary_failure_keeps_the_transcript_and_can_be_retried(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    session_id = new_session(engine, audio)
    transcriber = FakeTranscriber()

    def offline(_prompt: str) -> str:
        raise LLMError("could not connect to omlx at http://localhost:8000")

    failed = run(
        session_id, engine=engine, env=env, settings=settings, transcriber=transcriber, llm=FakeLLM(offline)
    )

    assert failed.status == SessionStatus.FAILED
    assert "could not connect to omlx" in failed.error
    assert len(rows(engine, TranscriptSegment, session_id)) == 1

    retried = run(
        session_id,
        engine=engine,
        env=env,
        settings=settings,
        transcriber=transcriber,
        llm=FakeLLM(),
        skip_transcription=True,
    )

    assert retried.status == SessionStatus.DONE and retried.error is None
    assert transcriber.calls == ["system-00000.wav"]  # not transcribed a second time


def test_provider_that_cannot_be_built_fails_the_session_instead_of_raising(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    session_id = new_session(engine, audio)

    def no_key(_settings, _env):
        raise LLMError("ANTHROPIC_API_KEY is not set")

    session = asyncio.run(
        process_session(
            session_id,
            engine=engine,
            env=env,
            settings=settings,
            transcriber_factory=lambda _s, _e: FakeTranscriber(),
            llm_factory=no_key,
            exporter_factory=lambda _s: None,
        )
    )

    assert session.status == SessionStatus.FAILED
    assert session.error == "ANTHROPIC_API_KEY is not set"


def test_session_without_speech_is_marked_empty_and_the_model_is_not_called(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "mic-00000.wav", 1.0, kind="silence")
    session_id = new_session(engine, audio)
    transcriber, llm = FakeTranscriber(), FakeLLM()

    session = run(session_id, engine=engine, env=env, settings=settings, transcriber=transcriber, llm=llm)

    assert session.status == SessionStatus.EMPTY
    assert transcriber.calls == [] and llm.prompts == []


def test_summarizing_again_keeps_finished_tasks_and_replaces_open_ones(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    session_id = new_session(engine, audio)
    transcriber = FakeTranscriber()
    run(session_id, engine=engine, env=env, settings=settings, transcriber=transcriber, llm=FakeLLM())

    with Session(engine) as db:
        done = db.exec(select(Task).where(Task.title == "Resolver a lista 3")).one()
        done.status = TaskStatus.DONE
        db.add(done)
        db.commit()

    second = json.dumps(
        {
            "title": "Derivadas",
            "summary": "nova versão",
            "tasks": [{"title": "resolver a lista 3!"}, {"title": "Assistir a aula 5"}],
        }
    )
    run(
        session_id,
        engine=engine,
        env=env,
        settings=settings,
        transcriber=transcriber,
        llm=FakeLLM(second),
        skip_transcription=True,
    )

    tasks = sorted(rows(engine, Task, session_id), key=lambda t: t.id)
    assert [(t.title, t.status) for t in tasks] == [
        ("Resolver a lista 3", TaskStatus.DONE),
        ("Assistir a aula 5", TaskStatus.OPEN),
    ]


def test_audio_is_deleted_after_success_when_not_kept(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    session_id = new_session(engine, audio)

    session = run(
        session_id,
        engine=engine,
        env=env,
        settings=settings.model_copy(update={"keep_audio": False}),
        transcriber=FakeTranscriber(),
        llm=FakeLLM(),
    )

    assert session.status == SessionStatus.DONE
    assert session.audio_dir is None and not audio.exists()


def test_deadlines_are_resolved_against_the_day_of_the_recording(tmp_path, engine, env, settings):
    from datetime import UTC, datetime

    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    with Session(engine) as db:
        old = StudySession(
            status=SessionStatus.RECORDED,
            audio_dir=str(audio),
            started_at=datetime(2026, 10, 5, 15, 0, tzinfo=UTC),
        )
        db.add(old)
        db.commit()
        session_id = old.id
    llm = FakeLLM()

    run(
        session_id,
        engine=engine,
        env=env,
        settings=settings,
        transcriber=FakeTranscriber(),
        llm=llm,
        today=None,
    )

    assert "Today is Monday, 2026-10-05" in llm.prompts[0]


def test_tasks_the_user_created_survive_a_new_summary(tmp_path, engine, env, settings):
    audio = tmp_path / "audio"
    make_wav(audio / "system-00000.wav", 1.0)
    session_id = new_session(engine, audio)
    transcriber = FakeTranscriber()
    run(session_id, engine=engine, env=env, settings=settings, transcriber=transcriber, llm=FakeLLM())
    with Session(engine) as db:
        db.add(Task(session_id=session_id, title="Perguntar ao professor sobre a prova"))
        db.commit()

    again = json.dumps({"title": "x", "summary": "novo", "tasks": [{"title": "Assistir a aula 5"}]})
    run(
        session_id,
        engine=engine,
        env=env,
        settings=settings,
        transcriber=transcriber,
        llm=FakeLLM(again),
        skip_transcription=True,
    )

    assert sorted(t.title for t in rows(engine, Task, session_id)) == [
        "Assistir a aula 5",
        "Perguntar ao professor sobre a prova",
    ]


def test_a_long_untimed_system_chunk_is_not_used_to_drop_mic_speech():
    words = "a derivada de uma função mede a taxa de variação instantânea"
    chunks = [
        TimedText("system", 0.0, 300.0, words + " e muito mais conteúdo da aula inteira"),
        TimedText("mic", 100.0, 105.0, words),
    ]

    assert drop_mic_echo(chunks) == chunks
