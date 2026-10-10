"""Interface for the apps that summaries and tasks are exported to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Literal

ReminderState = Literal["open", "done", "missing"]


class ExportError(RuntimeError):
    """The notes or reminders app could not be reached or refused the change."""


class Exporter(ABC):
    @abstractmethod
    def append_to_note(self, *, folder: str, name: str, section_html: str) -> str:
        """Appends to the note called ``name``, creating it (and the folder) if needed.

        Returns the identifier of the note.
        """

    @abstractmethod
    def write_note(self, *, folder: str, name: str, body_html: str, note_id: str | None = None) -> str:
        """Creates a note, or replaces the body of ``note_id`` if it still exists."""

    @abstractmethod
    def delete_note(self, note_id: str) -> None:
        """Deletes a note. Only used to clean up after the self-test."""

    @abstractmethod
    def create_reminder(
        self, *, list_name: str, title: str, notes: str, due: date | None, high_priority: bool
    ) -> str:
        """Creates a reminder (and the list if needed) and returns its identifier."""

    @abstractmethod
    def set_reminder_completed(self, reminder_id: str, completed: bool) -> None: ...

    @abstractmethod
    def delete_reminder(self, reminder_id: str) -> None: ...

    @abstractmethod
    def reminder_states(self, reminder_ids: list[str]) -> dict[str, ReminderState]:
        """State of each reminder. ``missing`` means the user deleted it."""
