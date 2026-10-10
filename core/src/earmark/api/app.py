"""HTTP API. Local only, and every route except /health needs the API token."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from .. import __version__, service
from ..config import UserSettings
from ..diagnostics import check_providers
from ..exporters import ExportError
from ..models import StudySession, Task, TaskStatus, TranscriptSegment
from ..recorder import RecorderError
from ..service import AppContext, InvalidState, NotFound, build_context
from .schemas import (
    Agenda,
    ImportRequest,
    RecordingStart,
    RecordingStarted,
    SessionDetail,
    SettingsUpdate,
    Status,
    SummarizeRequest,
    TaskCreate,
    TaskUpdate,
)


def create_app(ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or build_context(recover=True)
    token = ctx.env.resolve_api_token()
    jobs: dict[int, asyncio.Task] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # Never leave the helper recording after the server is gone.
        if ctx.active_session_id is not None:
            await asyncio.to_thread(service.stop_recording, ctx)
        for job in jobs.values():
            job.cancel()

    app = FastAPI(title="earmark", version=__version__, lifespan=lifespan)
    app.state.ctx = ctx

    def require_token(request: Request) -> None:
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid API token. Send 'Authorization: Bearer <token>' (see `earmark token`).",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(NotFound)
    async def not_found(_request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(RecorderError)
    @app.exception_handler(InvalidState)
    async def conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ExportError)
    async def export_failed(_request: Request, exc: ExportError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    api = APIRouter(dependencies=[Depends(require_token)])

    # -- status and settings ------------------------------------------------

    @api.get("/status", response_model=Status)
    def get_status() -> Status:
        return Status(
            version=__version__,
            recording=ctx.recorder.running,
            session_id=ctx.active_session_id,
            levels=ctx.recorder.levels if ctx.recorder.running else {},
            capture_available=ctx.recorder.available,
            processing=dict(ctx.progress),
        )

    @api.get("/providers")
    async def providers() -> dict:
        return await check_providers(ctx)

    @api.get("/settings", response_model=UserSettings)
    def get_settings() -> UserSettings:
        return ctx.settings_store.load()

    @api.put("/settings", response_model=UserSettings)
    def put_settings(body: SettingsUpdate) -> UserSettings:
        try:
            return ctx.settings_store.update(body.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # -- recording ----------------------------------------------------------

    @api.post("/recording/start", response_model=RecordingStarted, status_code=201)
    def start_recording(body: RecordingStart | None = None) -> RecordingStarted:
        session, info = service.start_recording(ctx, body.title if body else None)
        return RecordingStarted(session=session, mic=info.mic, system=info.system, warnings=info.warnings)

    @api.post("/recording/stop", response_model=SessionDetail)
    async def stop_recording(
        process: bool | None = Query(None, description="Defaults to the auto_process setting"),
        wait: bool = Query(False, description="Wait until processing has finished"),
    ) -> SessionDetail:
        session = await asyncio.to_thread(service.stop_recording, ctx)
        assert session.id is not None
        should_process = ctx.settings_store.load().auto_process if process is None else process
        if should_process:
            session = await launch(session.id, wait=wait)
        return detail(session)

    # -- sessions -----------------------------------------------------------

    @api.get("/sessions", response_model=list[StudySession])
    def list_sessions(limit: int = Query(50, ge=1, le=500)) -> list[StudySession]:
        return service.list_sessions(ctx, limit)

    @api.post("/sessions/import", response_model=SessionDetail, status_code=201)
    async def import_session(body: ImportRequest, process: bool = True, wait: bool = False) -> SessionDetail:
        session = await asyncio.to_thread(
            service.import_audio, ctx, Path(body.path), title=body.title, source=body.source
        )
        assert session.id is not None
        if process:
            session = await launch(session.id, wait=wait)
        return detail(session)

    @api.get("/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: int) -> SessionDetail:
        return detail(service.get_session(ctx, session_id))

    @api.get("/sessions/{session_id}/transcript", response_model=list[TranscriptSegment])
    def get_transcript(session_id: int) -> list[TranscriptSegment]:
        return service.get_transcript(ctx, session_id)

    @api.post("/sessions/{session_id}/process", response_model=SessionDetail)
    async def process_session(session_id: int, wait: bool = False) -> SessionDetail:
        """Transcribes the audio again and writes a new summary."""
        return detail(await launch(session_id, wait=wait))

    @api.post("/sessions/{session_id}/summarize", response_model=SessionDetail)
    async def summarize_session(
        session_id: int, body: SummarizeRequest | None = None, wait: bool = False
    ) -> SessionDetail:
        """Writes a new summary from the stored transcript, optionally with another model."""
        overrides = body.model_dump(exclude_none=True) if body else {}
        return detail(await launch(session_id, wait=wait, skip_transcription=True, overrides=overrides))

    @api.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: int) -> Response:
        if session_id in jobs and not jobs[session_id].done():
            raise HTTPException(409, "this session is being processed")
        service.delete_session(ctx, session_id)
        return Response(status_code=204)

    # -- tasks --------------------------------------------------------------

    @api.get("/tasks", response_model=list[Task])
    def list_tasks(
        task_status: TaskStatus | None = Query(TaskStatus.OPEN, alias="status"),
        session_id: int | None = None,
        include_all: bool = Query(False, alias="all", description="Ignore the status filter"),
    ) -> list[Task]:
        return service.list_tasks(ctx, status=None if include_all else task_status, session_id=session_id)

    @api.post("/tasks", response_model=Task, status_code=201)
    def create_task(body: TaskCreate) -> Task:
        if body.session_id is not None:
            service.get_session(ctx, body.session_id)
        return service.create_task(ctx, **body.model_dump())

    @api.patch("/tasks/{task_id}", response_model=Task)
    def update_task(task_id: int, body: TaskUpdate) -> Task:
        changes = body.model_dump(exclude_unset=True)
        # These cannot be cleared, only changed.
        for required in ("title", "priority", "status"):
            if changes.get(required, "") is None:
                del changes[required]
        return service.update_task(ctx, task_id, changes)

    @api.delete("/tasks/{task_id}", status_code=204)
    def delete_task(task_id: int) -> Response:
        service.delete_task(ctx, task_id)
        return Response(status_code=204)

    @api.post("/reminders/sync")
    async def sync_reminders() -> dict[str, int]:
        """Pulls completions and deletions made in the Reminders app."""
        return await asyncio.to_thread(service.sync_reminders, ctx)

    @api.get("/agenda", response_model=Agenda)
    def get_agenda(days: int = Query(7, ge=1, le=90)) -> Agenda:
        return Agenda(day=date.today(), days=days, **service.agenda(ctx, days=days))

    app.include_router(api)

    # -- helpers ------------------------------------------------------------

    def detail(session: StudySession) -> SessionDetail:
        assert session.id is not None
        return SessionDetail(
            session=session,
            tasks=service.list_tasks(ctx, status=None, session_id=session.id),
            progress=ctx.progress.get(session.id),
        )

    async def launch(
        session_id: int, *, wait: bool, skip_transcription: bool = False, overrides: dict | None = None
    ) -> StudySession:
        """Starts processing in the background, or runs it inline when ``wait`` is set."""
        running = jobs.get(session_id)
        if running is not None and not running.done():
            raise HTTPException(409, f"session {session_id} is already being processed")
        # Fail here, in the request, rather than inside a background job
        # nobody is waiting for.
        service.check_processable(ctx, session_id, skip_transcription=skip_transcription, overrides=overrides)

        job = asyncio.create_task(
            service.run_processing(
                ctx, session_id, skip_transcription=skip_transcription, overrides=overrides
            )
        )
        jobs[session_id] = job
        job.add_done_callback(
            lambda done: jobs.pop(session_id, None) if jobs.get(session_id) is done else None
        )
        if wait:
            return await job
        # Let the job mark the session as processing before answering.
        await asyncio.sleep(0)
        return service.get_session(ctx, session_id)

    return app
