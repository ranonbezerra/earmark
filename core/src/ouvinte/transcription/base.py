"""Common interface for speech to text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class TranscriptionError(RuntimeError):
    """The transcriber is unavailable or failed on a file."""


@dataclass(frozen=True)
class TranscriptChunk:
    """A stretch of speech. Times are seconds from the start of the file."""

    start_s: float
    end_s: float
    text: str


class Transcriber(ABC):
    name: str
    model: str

    @abstractmethod
    def transcribe(self, path: Path, language: str | None = None) -> list[TranscriptChunk]:
        """Blocking. The pipeline runs it on a dedicated worker thread."""

    def close(self) -> None:  # noqa: B027  (optional hook)
        pass


def chunks_from_response(payload: dict, fallback_duration: float | None) -> list[TranscriptChunk]:
    """Normalizes a Whisper style result (``text`` plus optional ``segments``)."""
    chunks = []
    for segment in payload.get("segments") or []:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        chunks.append(TranscriptChunk(start, float(segment.get("end", start)), text))
    if chunks:
        return chunks
    text = str(payload.get("text", "")).strip()
    if not text:
        return []
    duration = payload.get("duration") or fallback_duration or 0.0
    return [TranscriptChunk(0.0, float(duration), text)]
