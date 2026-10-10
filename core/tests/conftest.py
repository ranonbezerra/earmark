from __future__ import annotations

import json
import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from earmark.config import Env, SettingsStore, UserSettings
from earmark.db import make_engine
from earmark.exporters import Exporter, ExportError
from earmark.llm.base import LLMProvider
from earmark.recorder import Recorder, RecordingInfo
from earmark.service import AppContext
from earmark.transcription.base import Transcriber, TranscriptChunk

RATE = 16_000


def make_wav(path: Path, seconds: float = 1.0, *, kind: str = "tone", rate: int = RATE) -> Path:
    """Writes a 16-bit mono WAV with a 440 Hz tone or with silence."""
    frames = int(seconds * rate)
    if kind == "tone":
        samples = (int(12_000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
    else:
        samples = (0 for _ in range(frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return path


class FakeTranscriber(Transcriber):
    """Returns scripted chunks per file name and remembers what it was asked."""

    name = "fake-stt"
    model = "fake-whisper"

    def __init__(self, script: dict[str, list[TranscriptChunk]] | None = None):
        self.script = script or {}
        self.calls: list[str] = []
        self.closed = False

    def transcribe(self, path: Path, language: str | None = None) -> list[TranscriptChunk]:
        self.calls.append(path.name)
        return self.script.get(path.name, [TranscriptChunk(0.0, 1.0, f"words from {path.stem}")])

    def close(self) -> None:
        self.closed = True


SUMMARY_REPLY = json.dumps(
    {
        "title": "Derivatives",
        "summary": "## Derivatives\n\nThe derivative measures the rate of change.",
        "key_points": ["Chain rule"],
        "open_questions": ["When to use the product rule?"],
        "tasks": [
            {
                "title": "Finish problem set 3",
                "detail": "Exercises 1 to 10",
                "due_date": "2026-10-12",
                "priority": "high",
            },
            {"title": "Review limits", "detail": None, "due_date": None, "priority": "medium"},
        ],
    }
)


class FakeLLM(LLMProvider):
    """Replies with whatever ``reply`` returns and records every prompt."""

    name = "fake-llm"

    def __init__(self, reply: Callable[[str], str] | str = SUMMARY_REPLY, model: str = "fake-model"):
        self.model = model
        self._reply = reply
        self.prompts: list[str] = []
        self.closed = False

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        self.prompts.append(prompt)
        return self._reply(prompt) if callable(self._reply) else self._reply

    async def aclose(self) -> None:
        self.closed = True


class FakeRecorder(Recorder):
    """Stands in for the Swift helper: writes one segment per track on start."""

    def __init__(self, available: bool = True):
        super().__init__(["fake"] if available else None)
        self._on = False

    @property
    def running(self) -> bool:
        return self._on

    def start(self, out_dir: Path, *, mic=True, system=True, segment_seconds=300, ready_timeout=90.0):
        entries = []
        for track, enabled in (("system", system), ("mic", mic)):
            if enabled:
                make_wav(out_dir / f"{track}-00000.wav", 1.0)
                entries.append(
                    {"track": track, "index": 0, "file": f"{track}-00000.wav", "start_offset": 0.0}
                )
        (out_dir / "segments.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
        self._on = True
        self.levels = {"mic": -20.0, "system": -30.0}
        return RecordingInfo(mic=mic, system=system)

    def stop(self, timeout: float = 20.0) -> dict | None:
        self._on = False
        return {"event": "stopped", "duration": 1.0}


class FakeExporter(Exporter):
    """In-memory stand-in for the Notes and Reminders apps."""

    def __init__(self):
        self.notes: dict[str, dict] = {}  # id -> {"folder", "name", "body"}
        self.reminders: dict[str, dict] = {}  # id -> {"list", "title", "notes", "due", "high", "done"}
        self.fail: str | None = None
        self._next = 0

    def _id(self, kind: str) -> str:
        self._next += 1
        return f"{kind}-{self._next}"

    def _check(self) -> None:
        if self.fail:
            raise ExportError(self.fail)

    def append_to_note(self, *, folder, name, section_html):
        self._check()
        for note_id, note in self.notes.items():
            if (note["folder"], note["name"]) == (folder, name):
                note["body"] += section_html
                return note_id
        note_id = self._id("note")
        self.notes[note_id] = {"folder": folder, "name": name, "body": section_html}
        return note_id

    def write_note(self, *, folder, name, body_html, note_id=None):
        self._check()
        if note_id not in self.notes:
            note_id = self._id("note")
        self.notes[note_id] = {"folder": folder, "name": name, "body": body_html}
        return note_id

    def delete_note(self, note_id):
        self._check()
        self.notes.pop(note_id, None)

    def create_reminder(self, *, list_name, title, notes, due, high_priority):
        self._check()
        reminder_id = self._id("reminder")
        self.reminders[reminder_id] = {
            "list": list_name,
            "title": title,
            "notes": notes,
            "due": due,
            "high": high_priority,
            "done": False,
        }
        return reminder_id

    def set_reminder_completed(self, reminder_id, completed):
        self._check()
        self.reminders[reminder_id]["done"] = completed

    def delete_reminder(self, reminder_id):
        self._check()
        self.reminders.pop(reminder_id, None)

    def reminder_states(self, reminder_ids):
        self._check()
        return {
            rid: ("done" if self.reminders[rid]["done"] else "open") if rid in self.reminders else "missing"
            for rid in reminder_ids
        }


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    """No test may pick up real API keys, a real .env or the real data directory,
    and no test may ever write to the real Notes and Reminders apps."""
    monkeypatch.setattr("earmark.exporters.apple.is_available", lambda: False)
    for name in ("OMLX_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "EARMARK_COMPATIBLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EARMARK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def env(tmp_path) -> Env:
    env = Env(data_dir=tmp_path / "data", api_token="test-token")
    env.ensure_dirs()
    return env


@pytest.fixture
def engine(env):
    return make_engine(env.db_path)


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def exporter() -> FakeExporter:
    return FakeExporter()


@pytest.fixture
def ctx(env, engine, transcriber, llm, exporter) -> AppContext:
    return AppContext(
        env=env,
        engine=engine,
        settings_store=SettingsStore(env.settings_path),
        recorder=FakeRecorder(),
        transcriber_factory=lambda _settings, _env: transcriber,
        llm_factory=lambda _settings, _env: llm,
        exporter_factory=lambda _settings: exporter,
    )


@pytest.fixture
def settings() -> UserSettings:
    return UserSettings()
