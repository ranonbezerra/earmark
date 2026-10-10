"""Configuration.

Two layers on purpose:

* ``Env`` holds process level configuration and secrets. It comes from
  environment variables (or a ``.env`` file) and is never written to disk or
  returned by the API.
* ``UserSettings`` holds the choices the user makes at runtime (which model
  summarizes, which one transcribes, languages...). It is persisted as
  ``settings.json`` in the data directory and editable through the API.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["omlx", "openai", "anthropic", "openai-compatible"]
STTProviderName = Literal["mlx-whisper", "openai", "openai-compatible"]


def default_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "earmark"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "earmark"


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EARMARK_", env_file=".env", extra="ignore", populate_by_name=True
    )

    data_dir: Path = Field(default_factory=default_data_dir)
    host: str = "127.0.0.1"
    port: int = 8756
    capture_bin: Path | None = None
    api_token: str | None = None

    # Secrets. Read from the conventional variable names of each provider.
    omlx_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OMLX_API_KEY", "EARMARK_OMLX_API_KEY")
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "EARMARK_OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EARMARK_ANTHROPIC_API_KEY")
    )
    # Key for a generic OpenAI-compatible endpoint (LM Studio, Ollama, Groq...).
    compatible_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("EARMARK_COMPATIBLE_API_KEY")
    )

    @field_validator("data_dir", "capture_bin", mode="after")
    @classmethod
    def _expand_home(cls, value: Path | None) -> Path | None:
        # "~" is not expanded by the shell inside a .env file.
        return value.expanduser() if value is not None else None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "earmark.sqlite3"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "api-token"

    def ensure_dirs(self) -> None:
        # Recordings and transcripts are private: keep the folder owner-only.
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, mode=0o700)
            self.data_dir.chmod(0o700)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def resolve_api_token(self) -> str:
        """Token every API request must carry.

        The server only listens on localhost, but any web page open in a
        browser can still send requests to localhost. For something that can
        start a recording, that is not acceptable, so requests need a token
        that only local processes of this user can read.
        """
        if self.api_token:
            return self.api_token
        self.ensure_dirs()
        token = secrets.token_urlsafe(32)
        try:
            # Created atomically and owner-only; if another process got there
            # first, use its token instead.
            descriptor = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self.token_path.chmod(0o600)
            existing = self.token_path.read_text().strip()
            if existing:
                return existing
            self.token_path.write_text(token)
            return token
        with os.fdopen(descriptor, "w") as handle:
            handle.write(token)
        return token


class UserSettings(BaseModel):
    # Summaries and task extraction.
    llm_provider: LLMProviderName = "omlx"
    llm_model: str | None = None
    llm_base_url: str | None = None

    # Speech to text.
    stt_provider: STTProviderName = "mlx-whisper"
    stt_model: str | None = None
    stt_base_url: str | None = None

    # Spoken language as an ISO code ("pt", "en"). None lets the model detect it.
    language: str | None = None
    # Language of summaries and tasks. "auto" follows the transcript.
    summary_language: str = "auto"

    capture_mic: bool = True
    capture_system: bool = True
    segment_seconds: int = Field(default=300, ge=10, le=3600)

    # Segments whose loudest half second stays below this are not transcribed.
    silence_threshold_db: float = Field(default=-50.0, le=0)
    # Drop microphone text that just repeats what the speakers were playing.
    dedupe_mic_echo: bool = True
    # Transcript characters per LLM call. Longer sessions are summarized in
    # blocks and then consolidated, so small local context windows still work.
    chunk_chars: int = Field(default=12_000, ge=1_000)

    auto_process: bool = True
    keep_audio: bool = True

    # Apple Notes and Reminders (macOS only; ignored elsewhere).
    export_notes: bool = True
    export_reminders: bool = True
    notes_folder: str = Field(default="Earmark", min_length=1)
    reminders_list: str = Field(default="Earmark", min_length=1)
    # "day": one note per date, every session of that day appended to it.
    # "session": one note per session.
    notes_grouping: Literal["day", "session"] = "day"
    # Note titles are "<prefix> <date>", for example "Earmark 2026-10-10".
    note_title_prefix: str = "Earmark"
    # strftime format of the dates in note titles and due dates.
    note_date_format: str = "%Y-%m-%d"


class SettingsStore:
    """Loads and saves ``UserSettings`` as JSON."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> UserSettings:
        if not self.path.exists():
            return UserSettings()
        try:
            return UserSettings.model_validate(json.loads(self.path.read_text()))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid settings file {self.path}: {exc}") from exc

    def save(self, settings: UserSettings) -> UserSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(settings.model_dump(), indent=2) + "\n")
        return settings

    def update(self, changes: dict) -> UserSettings:
        merged = self.load().model_dump() | changes
        return self.save(UserSettings.model_validate(merged))
