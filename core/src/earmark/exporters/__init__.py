"""Sends summaries to a notes app and tasks to a reminders app.

The rendering and bookkeeping here do not know about Apple. ``AppleExporter``
is the only implementation for now.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import date

import markdown
from sqlalchemy import Engine
from sqlmodel import Session, select

from ..config import UserSettings
from ..models import StudySession, Task, TaskOrigin, TaskPriority, TaskStatus, utcnow
from . import apple
from .apple import AppleExporter
from .base import Exporter, ExportError

log = logging.getLogger(__name__)

__all__ = [
    "AppleExporter",
    "ExportError",
    "Exporter",
    "build_exporter",
    "export_session",
    "sync_reminders",
]


def build_exporter(settings: UserSettings) -> Exporter | None:
    """The exporter to use, or None when exporting is off or impossible here."""
    if not (settings.export_notes or settings.export_reminders):
        return None
    return AppleExporter() if apple.is_available() else None


UNTITLED = "Study session"


def note_name(session: StudySession, settings: UserSettings) -> str:
    """Title of the note a session goes to: the app's prefix plus the date."""
    day = session.started_at.astimezone().strftime(settings.note_date_format)
    name = f"{settings.note_title_prefix} {day}".strip()
    if settings.notes_grouping == "day":
        return name
    return f"{name} · {session.title or UNTITLED}"


_HEADING = re.compile(r"<(/?)h[1-6]>")


def render_section(
    session: StudySession, tasks: list[Task], settings: UserSettings, *, updated: bool = False
) -> str:
    """HTML for one session, in the small subset of HTML that Notes keeps."""
    started = session.started_at.astimezone()
    parts = []
    if settings.notes_grouping == "day":
        marker = " (updated)" if updated else ""
        title = html.escape(session.title or UNTITLED)
        parts.append(f"<h2>{started:%H:%M} · {title}{marker}</h2>")

    # Session notes are Markdown. Their headings become the smallest heading
    # level so they sit below the session title.
    body = markdown.markdown(session.summary or "", extensions=["sane_lists", "tables"])
    parts.append(_HEADING.sub(r"<\1h3>", body))

    for heading, items in (("Key points", session.key_points), ("Open questions", session.open_questions)):
        if items:
            rows = "".join(f"<li>{html.escape(item)}</li>" for item in items)
            parts.append(f"<h3>{heading}</h3><ul>{rows}</ul>")
    if tasks:
        rows = ""
        for task in tasks:
            due = f" (due {_day(task.due_date, settings)})" if task.due_date else ""
            rows += f"<li>{html.escape(task.title)}{due}</li>"
        parts.append(f"<h3>Tasks</h3><ul>{rows}</ul>")
    parts.append("<div><br></div>")
    return "".join(parts)


def _day(day: date, settings: UserSettings) -> str:
    return day.strftime(settings.note_date_format)


def export_session(session_id: int, *, engine: Engine, settings: UserSettings, exporter: Exporter) -> None:
    """Writes a summarized session to the notes app and its tasks to reminders.

    Raises ExportError. Whatever was exported before the error stays recorded,
    so running it again does not create duplicates of the reminders.
    """
    with Session(engine) as db:
        session = db.get(StudySession, session_id)
        if session is None or not session.summary:
            return
        tasks = list(
            db.exec(
                select(Task)
                .where(Task.session_id == session_id, Task.origin == TaskOrigin.EXTRACTED)
                .order_by(Task.id)  # type: ignore[arg-type]
            )
        )
        name = note_name(session, settings)

        if settings.export_notes:
            section = render_section(session, tasks, settings, updated=session.note_id is not None)
            if settings.notes_grouping == "day":
                session.note_id = exporter.append_to_note(
                    folder=settings.notes_folder, name=name, section_html=section
                )
            else:
                session.note_id = exporter.write_note(
                    folder=settings.notes_folder, name=name, body_html=section, note_id=session.note_id
                )
            session.note_name = name
            db.add(session)
            db.commit()

        if settings.export_reminders:
            day = _day(session.started_at.astimezone().date(), settings)
            context = f"Session: {session.title or UNTITLED} ({day})"
            if settings.export_notes:
                context += f"\nNote: {settings.notes_folder} / {name}"
            for task in tasks:
                if task.reminder_id or task.status != TaskStatus.OPEN:
                    continue
                task.reminder_id = exporter.create_reminder(
                    list_name=settings.reminders_list,
                    title=task.title,
                    notes=f"{task.detail}\n\n{context}" if task.detail else context,
                    due=task.due_date,
                    high_priority=task.priority == TaskPriority.HIGH,
                )
                db.add(task)
                db.commit()  # one by one: a failure later must not forget this id


def sync_reminders(*, engine: Engine, exporter: Exporter) -> dict[str, int]:
    """Brings changes made in the reminders app back.

    A reminder that was completed there completes the task here; one that was
    deleted there dismisses it. Raises ExportError if the app cannot be read,
    in which case nothing is changed.
    """
    with Session(engine) as db:
        tasks = list(
            db.exec(select(Task).where(Task.status == TaskStatus.OPEN, Task.reminder_id.is_not(None)))  # type: ignore[union-attr]
        )
        states = exporter.reminder_states([task.reminder_id for task in tasks if task.reminder_id])
        changed = {"completed": 0, "dismissed": 0}
        for task in tasks:
            state = states.get(task.reminder_id or "")
            if state == "done":
                task.status = TaskStatus.DONE
                task.completed_at = utcnow()
                changed["completed"] += 1
            elif state == "missing":
                task.status = TaskStatus.DISMISSED
                changed["dismissed"] += 1
            else:
                continue
            db.add(task)
        db.commit()
        return changed
