"""From recorded audio to a stored transcript, summary and task list."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, delete, select

from .audio import discover_segments, peak_window_dbfs, session_duration
from .config import Env, UserSettings
from .exporters import Exporter, ExportError, build_exporter, export_session
from .llm import LLMError, LLMProvider, build_llm
from .models import (
    SessionStatus,
    StudySession,
    Task,
    TaskOrigin,
    TaskPriority,
    TaskStatus,
    TranscriptSegment,
    utcnow,
)
from .summarizer import SessionSummary, Summarizer
from .transcription import Transcriber, TranscriptionError, build_transcriber

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
TranscriberFactory = Callable[[UserSettings, Env], Transcriber]
LLMFactory = Callable[[UserSettings, Env], LLMProvider]
ExporterFactory = Callable[[UserSettings], Exporter | None]

# A system chunk longer than this is a whole segment returned as one piece
# (some APIs give no timestamps). It says nothing about *when* words were
# played, so it cannot be used to decide that the microphone echoed them.
MAX_ECHO_REFERENCE_SECONDS = 45.0

# MLX keeps per-thread state, so every transcription call runs on this one
# thread. It also keeps two sessions from loading the model at the same time.
_STT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ouvinte-stt")

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TimedText:
    source: str
    start_s: float
    end_s: float
    text: str


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def transcript_lines(chunks: list[TimedText] | list[TranscriptSegment]) -> list[str]:
    return [f"[{format_timestamp(c.start_s)}] [{c.source}] {c.text}" for c in chunks]


def drop_mic_echo(
    chunks: list[TimedText], *, slack_s: float = 3.0, threshold: float = 0.8, min_words: int = 5
) -> list[TimedText]:
    """Removes microphone text that only repeats what the speakers were playing.

    When a lecture plays out loud, the microphone hears it too and the same
    sentences would show up twice. A mic chunk is dropped when most of its
    words appear in the system audio of the same moment. Short chunks are
    kept: a few words are not enough evidence.
    """
    system = [c for c in chunks if c.source == "system" and c.end_s - c.start_s <= MAX_ECHO_REFERENCE_SECONDS]
    kept = []
    for chunk in chunks:
        if chunk.source == "mic":
            words = _WORD.findall(chunk.text.lower())
            if len(words) >= min_words:
                nearby: set[str] = set()
                for other in system:
                    if other.start_s <= chunk.end_s + slack_s and other.end_s >= chunk.start_s - slack_s:
                        nearby.update(_WORD.findall(other.text.lower()))
                if nearby and sum(word in nearby for word in words) / len(words) >= threshold:
                    continue
        kept.append(chunk)
    return kept


async def transcribe_session(
    session_id: int,
    *,
    engine: Engine,
    settings: UserSettings,
    transcriber: Transcriber,
    progress: Progress | None = None,
) -> int:
    """Transcribes every non-silent segment and replaces the stored transcript.

    Returns the number of transcript rows written.
    """
    report = progress or (lambda _message: None)
    loop = asyncio.get_running_loop()
    with Session(engine) as db:
        session = db.get(StudySession, session_id)
        if session is None or not session.audio_dir:
            raise RuntimeError(f"session {session_id} has no audio to transcribe")
        audio_dir = Path(session.audio_dir)

    segments = discover_segments(audio_dir, settings.segment_seconds)
    if not segments:
        raise RuntimeError(f"no audio segments found in {audio_dir}")

    chunks: list[TimedText] = []
    for number, segment in enumerate(segments, start=1):
        report(f"transcribing {number}/{len(segments)}")
        if segment.duration_s is not None:
            level = await loop.run_in_executor(None, peak_window_dbfs, segment.path)
            if level is not None and level < settings.silence_threshold_db:
                continue
        found = await loop.run_in_executor(
            _STT_EXECUTOR, transcriber.transcribe, segment.path, settings.language
        )
        chunks.extend(
            TimedText(segment.source, segment.start_s + c.start_s, segment.start_s + c.end_s, c.text)
            for c in found
            if c.text.strip()
        )

    chunks.sort(key=lambda c: (c.start_s, c.source))
    if settings.dedupe_mic_echo:
        chunks = drop_mic_echo(chunks)

    with Session(engine) as db:
        db.exec(delete(TranscriptSegment).where(TranscriptSegment.session_id == session_id))  # type: ignore[arg-type]
        db.add_all(
            TranscriptSegment(
                session_id=session_id, source=c.source, start_s=c.start_s, end_s=c.end_s, text=c.text.strip()
            )
            for c in chunks
        )
        session = db.get(StudySession, session_id)
        assert session is not None
        session.duration_s = session.duration_s or session_duration(segments)
        session.stt_provider = transcriber.name
        session.stt_model = transcriber.model
        db.add(session)
        db.commit()
    return len(chunks)


async def summarize_session(
    session_id: int,
    *,
    engine: Engine,
    settings: UserSettings,
    llm: LLMProvider,
    progress: Progress | None = None,
    today: date | None = None,
) -> tuple[SessionSummary | None, list[str]]:
    """Summarizes the stored transcript.

    Returns the summary (None when there is no speech) and the reminder ids
    of tasks that the new summary replaced, so they can be removed from the
    reminders app.
    """
    with Session(engine) as db:
        session = db.get(StudySession, session_id)
        if session is None:
            raise LookupError(f"session {session_id} does not exist")
        # "Tomorrow" in a recording means the day after it was recorded, even
        # if the session is only summarized (or summarized again) days later.
        reference_day = today or session.started_at.astimezone().date()
        rows = db.exec(
            select(TranscriptSegment)
            .where(TranscriptSegment.session_id == session_id)
            .order_by(TranscriptSegment.start_s)  # type: ignore[arg-type]
        ).all()
        lines = transcript_lines(list(rows))
    if not lines:
        return None, []

    summarizer = Summarizer(
        llm,
        chunk_chars=settings.chunk_chars,
        language=settings.summary_language,
        today=reference_day,
        progress=progress,
    )
    summary = await summarizer.summarize(lines)
    model = await llm.resolved_model()

    with Session(engine) as db:
        session = db.get(StudySession, session_id)
        assert session is not None
        if summary.title and not session.title:
            session.title = summary.title
        session.summary = summary.summary
        session.key_points = summary.key_points
        session.open_questions = summary.open_questions
        session.language = summary.language or session.language
        session.llm_provider = llm.name
        session.llm_model = model
        db.add(session)
        stale_reminders = _replace_tasks(db, session_id, summary)
        db.commit()
    return summary, stale_reminders


def _replace_tasks(db: Session, session_id: int, summary: SessionSummary) -> list[str]:
    """Swaps the tasks a previous summary extracted for the new ones.

    Only open tasks that the model wrote are replaced. Tasks the user created,
    completed or dismissed are theirs: they stay, and a task that was already
    settled is not created again. Returns the reminder ids of removed tasks.
    """
    existing = db.exec(select(Task).where(Task.session_id == session_id)).all()
    kept = set()
    stale_reminders = []
    for task in existing:
        if task.status == TaskStatus.OPEN and task.origin == TaskOrigin.EXTRACTED:
            if task.reminder_id:
                stale_reminders.append(task.reminder_id)
            db.delete(task)
        else:
            kept.add(_normalize(task.title))
    for item in summary.tasks:
        if _normalize(item.title) in kept:
            continue
        kept.add(_normalize(item.title))
        db.add(
            Task(
                session_id=session_id,
                title=item.title,
                detail=item.detail,
                due_date=item.due_date,
                priority=TaskPriority(item.priority),
                origin=TaskOrigin.EXTRACTED,
            )
        )
    return stale_reminders


def _normalize(title: str) -> str:
    return " ".join(_WORD.findall(title.lower()))


async def process_session(
    session_id: int,
    *,
    engine: Engine,
    env: Env,
    settings: UserSettings,
    transcriber_factory: TranscriberFactory = build_transcriber,
    llm_factory: LLMFactory = build_llm,
    exporter_factory: ExporterFactory = build_exporter,
    skip_transcription: bool = False,
    progress: Progress | None = None,
    today: date | None = None,
) -> StudySession:
    """Runs the whole pipeline for one session and returns it.

    Failures do not raise: they are stored on the session (``status`` becomes
    ``failed`` and ``error`` says why). A transcript that was already stored
    survives a failed summary, so only the summary has to be retried.

    Exporting to the notes and reminders apps is best effort: if it fails the
    session is still ``done`` and ``export_error`` says what happened.
    """
    transcriber: Transcriber | None = None
    llm: LLMProvider | None = None
    _update(engine, session_id, status=SessionStatus.PROCESSING, error=None)
    try:
        if not skip_transcription:
            transcriber = transcriber_factory(settings, env)
            await transcribe_session(
                session_id, engine=engine, settings=settings, transcriber=transcriber, progress=progress
            )
        llm = llm_factory(settings, env)
        summary, stale_reminders = await summarize_session(
            session_id, engine=engine, settings=settings, llm=llm, progress=progress, today=today
        )
        status = SessionStatus.DONE if summary is not None else SessionStatus.EMPTY
        session = _update(engine, session_id, status=status, error=None, processed_at=utcnow())
        if status == SessionStatus.DONE:
            exporter = exporter_factory(settings)
            if exporter is not None:
                if progress:
                    progress("exporting to notes and reminders")
                problem = await asyncio.get_running_loop().run_in_executor(
                    None, _export, session_id, engine, settings, exporter, stale_reminders
                )
                session = _update(engine, session_id, export_error=problem)
        if status == SessionStatus.DONE and not settings.keep_audio and session.audio_dir:
            shutil.rmtree(session.audio_dir, ignore_errors=True)
            session = _update(engine, session_id, audio_dir=None)
        return session
    except (LLMError, TranscriptionError) as exc:
        # Expected ways to fail (server down, missing key, bad audio): no traceback needed.
        log.warning("processing session %s failed: %s", session_id, exc)
        return _update(engine, session_id, status=SessionStatus.FAILED, error=str(exc))
    except Exception as exc:  # noqa: BLE001  (anything that fails is reported on the session)
        log.exception("processing session %s failed", session_id)
        return _update(engine, session_id, status=SessionStatus.FAILED, error=str(exc) or type(exc).__name__)
    finally:
        if transcriber is not None:
            transcriber.close()
        if llm is not None:
            await llm.aclose()


def _export(
    session_id: int, engine: Engine, settings: UserSettings, exporter: Exporter, stale_reminders: list[str]
) -> str | None:
    """Runs the export and returns a description of what failed, if anything."""
    try:
        if settings.export_reminders:
            for reminder_id in stale_reminders:
                exporter.delete_reminder(reminder_id)
        export_session(session_id, engine=engine, settings=settings, exporter=exporter)
    except ExportError as exc:
        log.warning("exporting session %s failed: %s", session_id, exc)
        return str(exc)
    return None


def _update(engine: Engine, session_id: int, **changes) -> StudySession:
    with Session(engine) as db:
        session = db.get(StudySession, session_id)
        if session is None:
            raise LookupError(f"session {session_id} does not exist")
        for field, value in changes.items():
            setattr(session, field, value)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
