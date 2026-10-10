"""Command line interface. Works on the same data as the API, no server needed."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from . import __version__, service
from .audio import discover_segments, peak_window_dbfs
from .config import Env, UserSettings
from .diagnostics import check_providers
from .exporters import ExportError
from .models import SessionStatus, StudySession, Task, TaskStatus
from .pipeline import format_timestamp
from .recorder import RecorderError
from .service import AppContext, NotFound, build_context

PERMISSIONS_HINT = (
    "System Settings > Privacy & Security > Microphone, and > Screen & System Audio Recording "
    "(allow the app you launched earmark from, for example your terminal)"
)


def main(argv: list[str] | None = None) -> int:
    # Expected failures are printed by the commands themselves; only show
    # the log for genuine bugs.
    logging.basicConfig(level=logging.ERROR)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return args.handler(args) or 0
    except (NotFound, RecorderError, ExportError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earmark", description="Local-first study recorder for macOS.")
    parser.add_argument("--version", action="version", version=f"earmark {__version__}")
    commands = parser.add_subparsers(title="commands")

    def add(name: str, handler, summary: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=summary, description=summary)
        command.set_defaults(handler=handler)
        return command

    serve = add("serve", cmd_serve, "Run the local HTTP API.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    record = add("record", cmd_record, "Record a study session until Ctrl+C, then summarize it.")
    record.add_argument("--title")
    record.add_argument("--no-process", action="store_true", help="Only record; process later")

    imported = add("import", cmd_import, "Create a session from an existing audio file.")
    imported.add_argument("file", type=Path)
    imported.add_argument("--title")
    imported.add_argument("--source", choices=["system", "mic"], default="system")
    imported.add_argument("--no-process", action="store_true")

    process = add("process", cmd_process, "Transcribe and summarize a recorded session.")
    process.add_argument("session_id", type=int)
    process.add_argument("--summary-only", action="store_true", help="Reuse the stored transcript")

    add("sessions", cmd_sessions, "List sessions.")

    show = add("show", cmd_show, "Show the summary and tasks of a session.")
    show.add_argument("session_id", type=int)
    show.add_argument("--transcript", action="store_true")

    tasks = add("tasks", cmd_tasks, "Show what you have to do in the coming days.")
    tasks.add_argument("--days", type=int, default=7)

    add("sync", cmd_sync, "Pull completed and deleted reminders from the Reminders app.")

    done = add("done", cmd_done, "Mark tasks as done.")
    done.add_argument("task_ids", type=int, nargs="+")

    settings = add("settings", cmd_settings, "Show settings, or change them with key=value pairs.")
    settings.add_argument("pairs", nargs="*", metavar="key=value")

    doctor = add("doctor", cmd_doctor, "Check that capture, transcription and the summary model work.")
    doctor.add_argument("--capture-test", action="store_true", help="Record 5 seconds and report levels")
    doctor.add_argument(
        "--export-test",
        action="store_true",
        help="Write, check and remove a test note and a test reminder",
    )

    add("token", cmd_token, "Print the API token.")
    return parser


# -- commands ---------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    ctx = build_context(recover=True)
    host = args.host or ctx.env.host
    port = args.port or ctx.env.port
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"warning: listening on {host} exposes the recorder beyond this machine", file=sys.stderr)
    uvicorn.run(create_app(ctx), host=host, port=port, log_level="info")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    ctx = build_context()
    # Closing the terminal or a plain `kill` must end the recording as
    # cleanly as Ctrl+C does.
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, _raise_interrupt)

    print("Starting capture. The first time, macOS asks for permission.")
    try:
        session, info = service.start_recording(ctx, args.title)
    except KeyboardInterrupt:
        print("\nCancelled before the recording started.")
        return 130
    try:
        for warning in info.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        tracks = [name for name, on in (("microphone", info.mic), ("system audio", info.system)) if on]
        print(f"Recording session {session.id} ({' + '.join(tracks)}). Press Ctrl+C to stop.")
        started = time.monotonic()
        while ctx.recorder.running:
            time.sleep(1)
            levels = "  ".join(
                f"{track} {level:6.1f} dB" for track, level in sorted(ctx.recorder.levels.items())
            )
            print(f"\r  {format_timestamp(time.monotonic() - started)}  {levels}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        session = service.stop_recording(ctx)
    print(f"Stopped after {format_timestamp(session.duration_s or 0)}.")

    if args.no_process:
        print(f"Run `earmark process {session.id}` when you want the summary.")
        return 0
    return _process_and_print(ctx, session.id)


def _raise_interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def cmd_import(args: argparse.Namespace) -> int:
    ctx = build_context()
    session = service.import_audio(ctx, args.file, title=args.title, source=args.source)
    print(f"Imported as session {session.id}.")
    if args.no_process:
        return 0
    return _process_and_print(ctx, session.id)


def cmd_process(args: argparse.Namespace) -> int:
    ctx = build_context()
    service.get_session(ctx, args.session_id)
    return _process_and_print(ctx, args.session_id, skip_transcription=args.summary_only)


def cmd_sessions(_args: argparse.Namespace) -> int:
    ctx = build_context()
    sessions = service.list_sessions(ctx)
    if not sessions:
        print("No sessions yet. Start one with `earmark record`.")
    for session in sessions:
        duration = format_timestamp(session.duration_s) if session.duration_s else "--:--:--"
        print(
            f"{session.id:>4}  {session.started_at.astimezone():%Y-%m-%d %H:%M}  {duration}  "
            f"{session.status.value:<10}  {session.title or '(untitled)'}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    ctx = build_context()
    session = service.get_session(ctx, args.session_id)
    _print_session(ctx, session)
    if args.transcript:
        print("\n## Transcript\n")
        for row in service.get_transcript(ctx, args.session_id):
            print(f"[{format_timestamp(row.start_s)}] [{row.source}] {row.text}")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    ctx = build_context()
    _sync_quietly(ctx)
    groups = service.agenda(ctx, days=args.days)
    labels = {
        "overdue": "Overdue",
        "today": "Today",
        "upcoming": f"Next {args.days} days",
        "later": "Later",
        "unscheduled": "No date",
    }
    if not any(groups.values()):
        print("Nothing to do. Tasks appear here after a session is summarized.")
        return 0
    for key, label in labels.items():
        if groups[key]:
            print(f"\n{label}")
            for task in groups[key]:
                print(_task_line(task))
    return 0


def cmd_sync(_args: argparse.Namespace) -> int:
    ctx = build_context()
    changed = service.sync_reminders(ctx)
    print(f"{changed['completed']} completed and {changed['dismissed']} removed in Reminders.")
    return 0


def _sync_quietly(ctx: AppContext) -> None:
    """Picks up what was checked off in Reminders, without failing the command."""
    try:
        changed = service.sync_reminders(ctx)
    except ExportError as exc:
        print(f"warning: could not read Reminders ({exc})", file=sys.stderr)
        return
    if changed["completed"] or changed["dismissed"]:
        print(f"From Reminders: {changed['completed']} completed, {changed['dismissed']} removed.")


def cmd_done(args: argparse.Namespace) -> int:
    ctx = build_context()
    for task_id in args.task_ids:
        task = service.update_task(ctx, task_id, {"status": TaskStatus.DONE})
        print(f"done: {task.title}")
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    ctx = build_context()
    if args.pairs:
        fields = UserSettings.model_fields
        changes = {}
        for pair in args.pairs:
            key, separator, raw = pair.partition("=")
            if not separator or key not in fields:
                raise ValueError(f"expected key=value with a known key, got {pair!r}")
            try:
                changes[key] = json.loads(raw)  # numbers, true/false, null
            except ValueError:
                changes[key] = raw
        ctx.settings_store.update(changes)
    print(json.dumps(ctx.settings_store.load().model_dump(), indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ctx = build_context()
    report = asyncio.run(check_providers(ctx))
    healthy = True
    for name in ("capture", "stt", "llm"):
        item = report[name]
        healthy &= bool(item["available"])
        described = " ".join(str(item[key]) for key in ("provider", "model") if item.get(key))
        mark = "ok  " if item["available"] else "FAIL"
        print(f"[{mark}] {name:<8} {described}".rstrip())
        if item.get("detail"):
            print(f"       {item['detail']}")
    export = report["export"]
    if export["targets"]:
        mark = "ok  " if export["available"] else "FAIL"
        print(f"[{mark}] export   {' + '.join(export['targets'])}")
        if export.get("detail"):
            print(f"       {export['detail']}")
    print(f"data: {ctx.env.data_dir}")
    if args.capture_test:
        healthy &= _capture_test(ctx)
    if args.export_test:
        healthy &= _export_test(ctx)
    return 0 if healthy else 1


def cmd_token(_args: argparse.Namespace) -> int:
    print(Env().resolve_api_token())
    return 0


# -- helpers ----------------------------------------------------------------


def _process_and_print(ctx: AppContext, session_id: int, *, skip_transcription: bool = False) -> int:
    def progress(message: str) -> None:
        print(f"  {message}", flush=True)

    session = asyncio.run(
        service.run_processing(ctx, session_id, skip_transcription=skip_transcription, progress=progress)
    )
    if session.status == SessionStatus.FAILED:
        print(f"error: {session.error}", file=sys.stderr)
        if service.get_transcript(ctx, session_id):
            kept, retry = "recording and the transcript are", f"earmark process {session_id} --summary-only"
        else:
            kept, retry = "recording is", f"earmark process {session_id}"
        print(f"The {kept} safe. Fix the problem and run `{retry}`.", file=sys.stderr)
        return 1
    if session.status == SessionStatus.EMPTY:
        print("No speech was found in this session.")
        return 0
    print()
    _print_session(ctx, session)
    return 0


def _print_session(ctx: AppContext, session: StudySession) -> None:
    print(f"# {session.title or '(untitled)'}")
    print(f"session {session.id} · {session.started_at.astimezone():%Y-%m-%d %H:%M} · {session.status.value}")
    if session.error:
        print(f"error: {session.error}")
    if session.note_name:
        print(f"in Notes: {session.note_name}")
    if session.export_error:
        print(f"warning: not fully exported ({session.export_error})")
    if session.summary:
        print(f"\n{session.summary.strip()}")
    for heading, items in (("Key points", session.key_points), ("Open questions", session.open_questions)):
        if items:
            print(f"\n## {heading}\n")
            for item in items:
                print(f"- {item}")
    assert session.id is not None
    tasks = service.list_tasks(ctx, status=None, session_id=session.id)
    if tasks:
        print("\n## Tasks\n")
        for task in tasks:
            print(_task_line(task))


def _task_line(task: Task) -> str:
    box = {"open": "[ ]", "done": "[x]", "dismissed": "[-]"}[task.status.value]
    due = f"  (due {_describe_date(task.due_date)})" if task.due_date else ""
    urgent = " !" if task.priority.value == "high" else ""
    line = f"  {box} #{task.id} {task.title}{urgent}{due}"
    if task.detail:
        line += f"\n         {task.detail}"
    return line


def _describe_date(day: date) -> str:
    delta = (day - date.today()).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return f"{day:%a %Y-%m-%d}"


def _capture_test(ctx: AppContext, seconds: int = 5) -> bool:
    """Records a few seconds and reports whether each track actually has signal."""
    print(f"\nRecording {seconds}s. Play some audio and say something...")
    ok = True
    with tempfile.TemporaryDirectory(prefix="earmark-test-") as tmp:
        folder = Path(tmp)
        info = ctx.recorder.start(folder, mic=True, system=True, segment_seconds=60)
        try:
            time.sleep(seconds)
        finally:
            ctx.recorder.stop()
        for warning in info.warnings:
            print(f"  warning: {warning}")
        segments = {segment.source: segment for segment in discover_segments(folder, 60)}
        for track in ("mic", "system"):
            segment = segments.get(track)
            if segment is None:
                print(f"  [FAIL] {track:<7} nothing was recorded")
                ok = False
                continue
            level = peak_window_dbfs(segment.path)
            silent = level is None or level <= -90
            mark = "FAIL" if silent else "ok  "
            loudness = "unreadable" if level is None else f"loudest moment {level:.1f} dBFS"
            print(f"  [{mark}] {track:<7} {segment.duration_s or 0:.1f}s, {loudness}")
            if silent:
                ok = False
    if not ok:
        print(f"  Pure silence usually means a missing permission: {PERMISSIONS_HINT}.")
        print("  For system audio it can also mean nothing was playing during the test.")
    return ok


def _export_test(ctx: AppContext) -> bool:
    """Round trip through Notes and Reminders with throwaway items, then cleans up."""
    settings = ctx.settings_store.load().model_copy(update={"export_notes": True, "export_reminders": True})
    exporter = ctx.exporter_factory(settings)
    if exporter is None:
        print("\n  [FAIL] Notes and Reminders can only be reached on macOS")
        return False
    name = "earmark self-test"
    print("\nWriting a test note and a test reminder. macOS may ask for permission to control each app...")
    ok = True
    note_id = reminder_id = None
    try:
        note_id = exporter.append_to_note(folder=settings.notes_folder, name=name, section_html="<p>one</p>")
        again = exporter.append_to_note(folder=settings.notes_folder, name=name, section_html="<p>two</p>")
        same = note_id == again
        ok &= same
        print(f"  [{'ok  ' if same else 'FAIL'}] notes     a second write lands in the same note")

        reminder_id = exporter.create_reminder(
            list_name=settings.reminders_list,
            title=name,
            notes="Safe to delete.",
            due=date.today() + timedelta(days=1),
            high_priority=False,
        )
        exporter.set_reminder_completed(reminder_id, True)
        state = exporter.reminder_states([reminder_id]).get(reminder_id)
        good = state == "done"
        ok &= good
        print(f"  [{'ok  ' if good else 'FAIL'}] reminders created, completed and read back as {state!r}")
        exporter.delete_reminder(reminder_id)
        gone = exporter.reminder_states([reminder_id]).get(reminder_id) == "missing"
        reminder_id = None
        ok &= gone
        print(f"  [{'ok  ' if gone else 'FAIL'}] reminders a deleted reminder is seen as deleted")
    except ExportError as exc:
        print(f"  [FAIL] {exc}")
        ok = False
    finally:
        # Leave nothing behind, whatever happened above.
        for cleanup, item in ((exporter.delete_reminder, reminder_id), (exporter.delete_note, note_id)):
            if item:
                try:
                    cleanup(item)
                except ExportError:
                    print(f"  could not remove the test item {name!r}; delete it by hand", file=sys.stderr)
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
