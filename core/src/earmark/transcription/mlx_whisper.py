"""On-device transcription with Whisper on MLX (Apple Silicon)."""

from __future__ import annotations

import wave
from pathlib import Path

from .base import Transcriber, TranscriptChunk, TranscriptionError, chunks_from_response

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_SAMPLE_RATE = 16_000


def is_available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


class MlxWhisperTranscriber(Transcriber):
    name = "mlx-whisper"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODEL

    def transcribe(self, path: Path, language: str | None = None) -> list[TranscriptChunk]:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise TranscriptionError(
                "mlx-whisper is not installed. Run `uv sync --extra local` on an Apple Silicon Mac, "
                "or switch stt_provider to an API."
            ) from exc

        # The capture helper already writes 16 kHz mono, so the samples can go
        # straight to the model. Anything else is decoded by mlx-whisper itself,
        # which needs ffmpeg installed.
        audio = _load_native_wav(path)
        try:
            result = mlx_whisper.transcribe(
                audio if audio is not None else str(path),
                path_or_hf_repo=self.model,
                language=language,
                # Feeding previous text back in is what makes Whisper loop on
                # the same sentence across long, quiet recordings.
                condition_on_previous_text=False,
                verbose=None,
            )
        except Exception as exc:  # noqa: BLE001  (model and decoding errors vary)
            raise TranscriptionError(f"mlx-whisper failed on {path.name}: {exc}") from exc
        return chunks_from_response(result, fallback_duration=None)


def _load_native_wav(path: Path):
    """Samples as float32 in [-1, 1] if the file is 16 kHz mono 16-bit WAV."""
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (WHISPER_SAMPLE_RATE, 1, 2):
                return None
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    import numpy as np

    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
