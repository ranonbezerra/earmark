"""Apple Notes and Reminders, driven through AppleScript.

Neither app has a public API that Python can call, so this talks to them with
``osascript``. Every value is passed as an argument to ``on run argv``: text
from a transcript is never spliced into the script source.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from datetime import date

from .base import Exporter, ExportError, ReminderState

OSASCRIPT = "/usr/bin/osascript"

_ENSURE_FOLDER = """
        if not (exists folder folderName) then
            make new folder with properties {name:folderName}
        end if
"""

_APPEND_TO_NOTE = f"""
on run argv
    set folderName to item 1 of argv
    set noteName to item 2 of argv
    set titleHtml to item 3 of argv
    set sectionHtml to item 4 of argv
    tell application "Notes"
{_ENSURE_FOLDER}
        set theFolder to folder folderName
        set found to (every note of theFolder whose name is noteName)
        if (count of found) is 0 then
            set theNote to make new note at theFolder with properties {{body:titleHtml & sectionHtml}}
        else
            set theNote to item 1 of found
            set body of theNote to (body of theNote) & sectionHtml
        end if
        return id of theNote
    end tell
end run
"""

_WRITE_NOTE = f"""
on run argv
    set folderName to item 1 of argv
    set noteId to item 2 of argv
    set bodyHtml to item 3 of argv
    tell application "Notes"
{_ENSURE_FOLDER}
        if noteId is not "" then
            try
                set theNote to note id noteId
                set body of theNote to bodyHtml
                return id of theNote
            on error number -1728
                -- the note was deleted: fall through and create a new one
            end try
        end if
        set theNote to make new note at folder folderName with properties {{body:bodyHtml}}
        return id of theNote
    end tell
end run
"""

_DELETE_NOTE = """
on run argv
    tell application "Notes"
        try
            delete (note id (item 1 of argv))
        on error number -1728
            -- already gone
        end try
    end tell
end run
"""

# The date is assembled from numbers because AppleScript parses date strings
# according to the system language.
_CREATE_REMINDER = """
on run argv
    set listName to item 1 of argv
    set theTitle to item 2 of argv
    set theNotes to item 3 of argv
    set dueYear to item 4 of argv
    set dueMonth to item 5 of argv
    set dueDay to item 6 of argv
    set highPriority to item 7 of argv
    tell application "Reminders"
        if not (exists list listName) then
            make new list with properties {name:listName}
        end if
        tell list listName
            set theReminder to make new reminder with properties {name:theTitle, body:theNotes}
        end tell
        if dueYear is not "" then
            set theDate to current date
            set time of theDate to 0
            set day of theDate to 1
            set year of theDate to (dueYear as integer)
            set month of theDate to (dueMonth as integer)
            set day of theDate to (dueDay as integer)
            set allday due date of theReminder to theDate
        end if
        if highPriority is "1" then
            set priority of theReminder to 1
        end if
        return id of theReminder
    end tell
end run
"""

_SET_COMPLETED = """
on run argv
    tell application "Reminders"
        set completed of (reminder id (item 1 of argv)) to ((item 2 of argv) is "1")
    end tell
end run
"""

_DELETE_REMINDER = """
on run argv
    tell application "Reminders"
        try
            delete (reminder id (item 1 of argv))
        on error number -1728
            -- already gone
        end try
    end tell
end run
"""

# Only "no such object" (-1728) counts as missing. Any other error, such as a
# denied permission, aborts the script so nothing is mistaken for a deletion.
_REMINDER_STATES = """
on run argv
    set output to ""
    tell application "Reminders"
        count of lists
        repeat with reminderId in argv
            set state to "missing"
            try
                if completed of (reminder id (reminderId as text)) then
                    set state to "done"
                else
                    set state to "open"
                end if
            on error number -1728
                set state to "missing"
            end try
            set output to output & (reminderId as text) & tab & state & linefeed
        end repeat
    end tell
    return output
end run
"""


def is_available() -> bool:
    return sys.platform == "darwin" and shutil.which(OSASCRIPT) is not None


class AppleExporter(Exporter):
    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout

    def append_to_note(self, *, folder: str, name: str, section_html: str) -> str:
        # Notes takes the first line of the body as the note's name.
        title_html = f"<h1>{html.escape(name)}</h1>"
        return self._run("Notes", _APPEND_TO_NOTE, folder, name, title_html, section_html)

    def write_note(self, *, folder: str, name: str, body_html: str, note_id: str | None = None) -> str:
        full = f"<h1>{html.escape(name)}</h1>{body_html}"
        return self._run("Notes", _WRITE_NOTE, folder, note_id or "", full)

    def delete_note(self, note_id: str) -> None:
        self._run("Notes", _DELETE_NOTE, note_id)

    def create_reminder(
        self, *, list_name: str, title: str, notes: str, due: date | None, high_priority: bool
    ) -> str:
        due_parts = (str(due.year), str(due.month), str(due.day)) if due else ("", "", "")
        return self._run(
            "Reminders", _CREATE_REMINDER, list_name, title, notes, *due_parts, "1" if high_priority else "0"
        )

    def set_reminder_completed(self, reminder_id: str, completed: bool) -> None:
        self._run("Reminders", _SET_COMPLETED, reminder_id, "1" if completed else "0")

    def delete_reminder(self, reminder_id: str) -> None:
        self._run("Reminders", _DELETE_REMINDER, reminder_id)

    def reminder_states(self, reminder_ids: list[str]) -> dict[str, ReminderState]:
        if not reminder_ids:
            return {}
        states: dict[str, ReminderState] = {}
        for line in self._run("Reminders", _REMINDER_STATES, *reminder_ids).splitlines():
            reminder_id, _, state = line.partition("\t")
            if state in ("open", "done", "missing"):
                states[reminder_id] = state  # type: ignore[assignment]
        return states

    def _run(self, app: str, script: str, *arguments: str) -> str:
        try:
            result = subprocess.run(  # noqa: S603  (fixed binary, values passed as arguments)
                [OSASCRIPT, "-", *arguments],
                input=script,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExportError(
                f"{app} did not answer in {self._timeout:.0f}s. If macOS is asking for permission "
                f"to control {app}, answer the prompt and try again."
            ) from exc
        except OSError as exc:
            raise ExportError(f"could not run osascript: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            if "-1743" in detail:
                detail = (
                    f"not allowed to control {app}. Allow it in System Settings > Privacy & Security "
                    "> Automation, for the app you launched earmark from."
                )
            raise ExportError(f"{app}: {detail}")
        return result.stdout.strip()
