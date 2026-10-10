"""Operations shared by the HTTP API and the command line."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, delete, select

from .config import Env, SettingsStore, UserSettings
from .db import make_engine, recover_interrupted
from .exporters import Exporter, ExportError, build_exporter
from .exporters import sync_reminders as _sync_reminders
from .llm import build_llm
from .models import SessionStatus, StudySession, Task, TaskStatus, TranscriptSegment, utcnow
from .pipeline import ExporterFactory, LLMFactory, Progress, TranscriberFactory, process_session
from .recorder import Recorder, RecorderError, RecordingInfo
from .transcription import build_transcriber

log = logging.getLogger(__name__)


class NotFound(LookupError):
    pass


class InvalidState(RuntimeError):
    """The request makes no sense for the session as it is right now."""


@dataclass
class AppContext:
    env: Env
    engine: Engine
    settings_store: SettingsStore
    recorder: Recorder
    transcriber_factory: TranscriberFactory = build_transcriber
    llm_factory: LLMFactory = build_llm
    exporter_factory: ExporterFactory = build_exporter
    active_session_id: int | None = None
    # Human readable stage of each session being processed right now.
    progress: dict[int, str] = field(default_factory=dict)


def build_context(env: Env | None = None, *, recover: bool = False) -> AppContext:
    """Opens the data directory.

    ``recover`` resets sessions that a crash left as recording or processing.
    Only the long-running server asks for it: a one-off command must not
    touch a session that the server is recording at that very moment.
    """
    env = env or Env()
    env.ensure_dirs()
    engine = make_engine(env.db_path)
    if recover:
        recover_interrupted(engine)
    return AppContext(
        env=env,
        engine=engine,
        settings_store=SettingsStore(env.settings_path),
        recorder=Recorder.from_env(env),
    )


# -- sessions ---------------------------------------------------------------


def get_session(ctx: AppContext, session_id: int) -> StudySession:
    with Session(ctx.engine) as db:
        session = db.get(StudySession, session_id)
    if session is None:
        raise NotFound(f"session {session_id} does not exist")
    return session


def list_sessions(ctx: AppContext, limit: int = 50) -> list[StudySession]:
    with Session(ctx.engine) as db:
        return list(
            db.exec(select(StudySession).order_by(StudySession.started_at.desc()).limit(limit))  # type: ignore[attr-defined]
        )


def get_transcript(ctx: AppContext, session_id: int) -> list[TranscriptSegment]:
    get_session(ctx, session_id)
    with Session(ctx.engine) as db:
        return list(
            db.exec(
                select(TranscriptSegment)
                .where(TranscriptSegment.session_id == session_id)
                .order_by(TranscriptSegment.start_s)  # type: ignore[arg-type]
            )
        )


def start_recording(ctx: AppContext, title: str | None = None) -> tuple[StudySession, RecordingInfo]:
    if ctx.recorder.running or ctx.active_session_id is not None:
        raise RecorderError("a recording is already in progress")
    settings = ctx.settings_store.load()

    with Session(ctx.engine) as db:
        session = StudySession(title=title or None, status=SessionStatus.RECORDING)
        db.add(session)
        db.commit()
        db.refresh(session)
        assert session.id is not None
        audio_dir = ctx.env.sessions_dir / f"{session.id:06d}"
        session.audio_dir = str(audio_dir)
        db.add(session)
        db.commit()
        db.refresh(session)

    # Claimed before the helper starts: starting can take a while (permission
    # prompts) and nothing else may touch this session in the meantime.
    ctx.active_session_id = session.id
    try:
        info = ctx.recorder.start(
            audio_dir,
            mic=settings.capture_mic,
            system=settings.capture_system,
            segment_seconds=settings.segment_seconds,
        )
    except BaseException:
        # Nothing was recorded: do not leave an empty session behind.
        ctx.active_session_id = None
        delete_session(ctx, session.id)
        raise
    return session, info


def stop_recording(ctx: AppContext) -> StudySession:
    session_id = ctx.active_session_id
    if session_id is None:
        raise RecorderError("no recording in progress")
    stopped = ctx.recorder.stop()
    ctx.active_session_id = None
    with Session(ctx.engine) as db:
        session = db.get(StudySession, session_id)
        if session is None:
            raise NotFound(f"session {session_id} does not exist")
        session.status = SessionStatus.RECORDED
        session.ended_at = utcnow()
        if stopped and isinstance(stopped.get("duration"), int | float):
            session.duration_s = float(stopped["duration"])
        else:
            session.duration_s = (session.ended_at - session.started_at).total_seconds()
        db.add(session)
        db.commit()
        db.refresh(session)
        return session


def import_audio(
    ctx: AppContext, path: Path, *, title: str | None = None, source: str = "system"
) -> StudySession:
    """Creates a session from an existing audio file (copied, never moved)."""
    if source not in {"system", "mic"}:
        raise ValueError("source must be 'system' or 'mic'")
    path = path.expanduser()
    if not path.is_file():
        raise NotFound(f"{path} is not a file")
    with Session(ctx.engine) as db:
        session = StudySession(title=title or path.stem, status=SessionStatus.RECORDED, ended_at=utcnow())
        db.add(session)
        db.commit()
        db.refresh(session)
        assert session.id is not None
        audio_dir = ctx.env.sessions_dir / f"{session.id:06d}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower() if re.fullmatch(r"\.[a-z0-9]+", path.suffix.lower()) else ".wav"
        shutil.copy2(path, audio_dir / f"{source}-00000{suffix}")
        session.audio_dir = str(audio_dir)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session


def delete_session(ctx: AppContext, session_id: int) -> None:
    if ctx.active_session_id == session_id:
        raise RecorderError("stop the recording before deleting its session")
    with Session(ctx.engine) as db:
        session = db.get(StudySession, session_id)
        if session is None:
            raise NotFound(f"session {session_id} does not exist")
        audio_dir = session.audio_dir
        reminders = [
            task.reminder_id
            for task in db.exec(select(Task).where(Task.session_id == session_id))
            if task.reminder_id and task.status == TaskStatus.OPEN
        ]
        db.exec(delete(TranscriptSegment).where(TranscriptSegment.session_id == session_id))  # type: ignore[arg-type]
        db.exec(delete(Task).where(Task.session_id == session_id))  # type: ignore[arg-type]
        db.delete(session)
        db.commit()
    if audio_dir:
        # Only ever remove directories this app created.
        target = Path(audio_dir).resolve()
        if target.parent == ctx.env.sessions_dir.resolve():
            shutil.rmtree(target, ignore_errors=True)
    for reminder_id in reminders:
        _with_exporter(ctx, lambda exporter, rid=reminder_id: exporter.delete_reminder(rid))


def check_processable(
    ctx: AppContext, session_id: int, *, skip_transcription: bool = False, overrides: dict | None = None
) -> UserSettings:
    """Refuses requests that cannot work, before any background work starts.

    Returns the settings the run will use.
    """
    session = get_session(ctx, session_id)
    if session.status == SessionStatus.RECORDING and ctx.active_session_id == session_id:
        raise InvalidState(f"session {session_id} is still being recorded")
    if skip_transcription and not get_transcript(ctx, session_id):
        raise InvalidState(f"session {session_id} has no transcript yet; process it first")
    settings = ctx.settings_store.load()
    if overrides:
        settings = UserSettings.model_validate(settings.model_dump() | overrides)
    return settings


async def run_processing(
    ctx: AppContext,
    session_id: int,
    *,
    skip_transcription: bool = False,
    overrides: dict | None = None,
    progress: Progress | None = None,
) -> StudySession:
    """Processes a session with the current settings (plus one-off overrides)."""
    settings = check_processable(ctx, session_id, skip_transcription=skip_transcription, overrides=overrides)

    def report(message: str) -> None:
        ctx.progress[session_id] = message
        if progress:
            progress(message)

    try:
        return await process_session(
            session_id,
            engine=ctx.engine,
            env=ctx.env,
            settings=settings,
            transcriber_factory=ctx.transcriber_factory,
            llm_factory=ctx.llm_factory,
            exporter_factory=ctx.exporter_factory,
            skip_transcription=skip_transcription,
            progress=report,
        )
    finally:
        ctx.progress.pop(session_id, None)


# -- tasks ------------------------------------------------------------------


def list_tasks(
    ctx: AppContext, *, status: TaskStatus | None = TaskStatus.OPEN, session_id: int | None = None
) -> list[Task]:
    query = select(Task)
    if status is not None:
        query = query.where(Task.status == status)
    if session_id is not None:
        query = query.where(Task.session_id == session_id)
    with Session(ctx.engine) as db:
        tasks = list(db.exec(query))
    # Dated tasks first, soonest on top; undated ones after, newest first.
    return sorted(tasks, key=lambda t: (t.due_date is None, t.due_date or date.max, -(t.id or 0)))


def create_task(ctx: AppContext, **fields) -> Task:
    with Session(ctx.engine) as db:
        task = Task(**fields)
        db.add(task)
        db.commit()
        db.refresh(task)
        return task


def update_task(ctx: AppContext, task_id: int, changes: dict) -> Task:
    with Session(ctx.engine) as db:
        task = db.get(Task, task_id)
        if task is None:
            raise NotFound(f"task {task_id} does not exist")
        was_done = task.status == TaskStatus.DONE
        for name, value in changes.items():
            setattr(task, name, value)
        if "status" in changes:
            task.completed_at = utcnow() if task.status == TaskStatus.DONE else None
        db.add(task)
        db.commit()
        db.refresh(task)
    is_done = task.status == TaskStatus.DONE
    if task.reminder_id and is_done != was_done:
        # Keep the reminders app in step. Best effort: the task is the record.
        reminder_id = task.reminder_id
        _with_exporter(ctx, lambda exporter: exporter.set_reminder_completed(reminder_id, is_done))
    return task


def delete_task(ctx: AppContext, task_id: int) -> None:
    with Session(ctx.engine) as db:
        task = db.get(Task, task_id)
        if task is None:
            raise NotFound(f"task {task_id} does not exist")
        reminder_id = task.reminder_id
        db.delete(task)
        db.commit()
    if reminder_id:
        _with_exporter(ctx, lambda exporter: exporter.delete_reminder(reminder_id))


def sync_reminders(ctx: AppContext) -> dict[str, int]:
    """Pulls completions and deletions made in the reminders app.

    Returns how many tasks changed. Raises ExportError if the app cannot be
    read; with exporting turned off it does nothing.
    """
    settings = ctx.settings_store.load()
    exporter = ctx.exporter_factory(settings) if settings.export_reminders else None
    if exporter is None:
        return {"completed": 0, "dismissed": 0}
    return _sync_reminders(engine=ctx.engine, exporter=exporter)


def _with_exporter(ctx: AppContext, action) -> None:
    """Runs a small follow-up change in the reminders app, never failing the caller."""
    try:
        settings = ctx.settings_store.load()
        exporter: Exporter | None = ctx.exporter_factory(settings) if settings.export_reminders else None
        if exporter is not None:
            action(exporter)
    except (ExportError, RuntimeError) as exc:
        log.warning("could not update the reminders app: %s", exc)


def agenda(ctx: AppContext, *, days: int = 7, today: date | None = None) -> dict[str, list[Task]]:
    """Open tasks grouped by how soon they are due."""
    today = today or date.today()
    horizon = today + timedelta(days=days)
    groups: dict[str, list[Task]] = {
        "overdue": [],
        "today": [],
        "upcoming": [],
        "later": [],
        "unscheduled": [],
    }
    for task in list_tasks(ctx, status=TaskStatus.OPEN):
        if task.due_date is None:
            groups["unscheduled"].append(task)
        elif task.due_date < today:
            groups["overdue"].append(task)
        elif task.due_date == today:
            groups["today"].append(task)
        elif task.due_date <= horizon:
            groups["upcoming"].append(task)
        else:
            groups["later"].append(task)
    return groups
