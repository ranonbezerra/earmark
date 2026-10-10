"""Request and response bodies of the HTTP API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from ..config import LLMProviderName, STTProviderName
from ..models import StudySession, Task, TaskPriority, TaskStatus


class RecordingStart(BaseModel):
    title: str | None = None


class RecordingStarted(BaseModel):
    session: StudySession
    mic: bool
    system: bool
    warnings: list[str] = []


class ImportRequest(BaseModel):
    path: str
    title: str | None = None
    source: Literal["system", "mic"] = "system"


class SessionDetail(BaseModel):
    session: StudySession
    tasks: list[Task]
    progress: str | None = None


class SummarizeRequest(BaseModel):
    """One-off overrides, to try another model without changing the settings."""

    llm_provider: LLMProviderName | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    summary_language: str | None = None


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    detail: str | None = None
    due_date: date | None = None
    priority: TaskPriority = TaskPriority.MEDIUM
    session_id: int | None = None


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    detail: str | None = None
    due_date: date | None = None
    priority: TaskPriority | None = None
    status: TaskStatus | None = None


class SettingsUpdate(BaseModel):
    llm_provider: LLMProviderName | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    stt_provider: STTProviderName | None = None
    stt_model: str | None = None
    stt_base_url: str | None = None
    language: str | None = None
    summary_language: str | None = None
    capture_mic: bool | None = None
    capture_system: bool | None = None
    segment_seconds: int | None = None
    silence_threshold_db: float | None = None
    dedupe_mic_echo: bool | None = None
    chunk_chars: int | None = None
    auto_process: bool | None = None
    keep_audio: bool | None = None
    export_notes: bool | None = None
    export_reminders: bool | None = None
    notes_folder: str | None = None
    reminders_list: str | None = None
    notes_grouping: Literal["day", "session"] | None = None
    note_title_prefix: str | None = None
    note_date_format: str | None = None


class Agenda(BaseModel):
    day: date
    days: int
    overdue: list[Task]
    today: list[Task]
    upcoming: list[Task]
    later: list[Task]
    unscheduled: list[Task]


class Status(BaseModel):
    version: str
    recording: bool
    session_id: int | None
    levels: dict[str, float]
    capture_available: bool
    processing: dict[int, str]
