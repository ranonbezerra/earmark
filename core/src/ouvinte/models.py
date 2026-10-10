"""Database tables."""

from datetime import UTC, date, datetime
from enum import StrEnum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    # Timestamps are timezone aware and stored in UTC; convert at the edges.
    return datetime.now(UTC)


class SessionStatus(StrEnum):
    RECORDING = "recording"
    RECORDED = "recorded"
    PROCESSING = "processing"
    DONE = "done"
    EMPTY = "empty"  # processed, but nothing was said
    FAILED = "failed"


class TaskStatus(StrEnum):
    OPEN = "open"
    DONE = "done"
    DISMISSED = "dismissed"


class TaskOrigin(StrEnum):
    EXTRACTED = "extracted"  # written by the summary model
    MANUAL = "manual"  # created by the user


class TaskPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StudySession(SQLModel, table=True):
    __tablename__ = "study_session"

    id: int | None = Field(default=None, primary_key=True)
    title: str | None = None
    status: SessionStatus = Field(default=SessionStatus.RECORDING, index=True)
    error: str | None = None

    started_at: datetime = Field(default_factory=utcnow, index=True)
    ended_at: datetime | None = None
    duration_s: float | None = None
    audio_dir: str | None = None

    summary: str | None = None
    key_points: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    open_questions: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # What produced the transcript and the summary, for traceability.
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    processed_at: datetime | None = None
    # Language the summary was written in ("pt", "en"...), as reported by the model.
    language: str | None = None

    # Where the summary went in the Notes app, and what went wrong if it did not.
    note_id: str | None = None
    note_name: str | None = None
    export_error: str | None = None


class TranscriptSegment(SQLModel, table=True):
    __tablename__ = "transcript_segment"

    id: int | None = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="study_session.id", index=True)
    source: str  # "mic" or "system"
    start_s: float
    end_s: float
    text: str


class Task(SQLModel, table=True):
    __tablename__ = "task"

    id: int | None = Field(default=None, primary_key=True)
    session_id: int | None = Field(default=None, foreign_key="study_session.id", index=True)
    title: str
    detail: str | None = None
    due_date: date | None = Field(default=None, index=True)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = Field(default=TaskStatus.OPEN, index=True)
    origin: TaskOrigin = TaskOrigin.MANUAL
    # Identifier of the matching item in the Reminders app, once exported.
    reminder_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
