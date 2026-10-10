"""Speech to text providers and the factory that picks one from settings."""

from __future__ import annotations

from ..config import Env, UserSettings
from .base import Transcriber, TranscriptChunk, TranscriptionError
from .mlx_whisper import MlxWhisperTranscriber
from .openai_compat import OpenAICompatTranscriber

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "whisper-1"

__all__ = [
    "MlxWhisperTranscriber",
    "OpenAICompatTranscriber",
    "Transcriber",
    "TranscriptChunk",
    "TranscriptionError",
    "build_transcriber",
]


def build_transcriber(settings: UserSettings, env: Env) -> Transcriber:
    provider = settings.stt_provider

    if provider == "mlx-whisper":
        return MlxWhisperTranscriber(settings.stt_model)

    if provider == "openai":
        if not env.openai_api_key:
            raise TranscriptionError("OPENAI_API_KEY is not set")
        return OpenAICompatTranscriber(
            name="openai",
            base_url=settings.stt_base_url or OPENAI_BASE_URL,
            model=settings.stt_model or OPENAI_DEFAULT_MODEL,
            api_key=env.openai_api_key,
        )

    if provider == "openai-compatible":
        if not settings.stt_base_url or not settings.stt_model:
            raise TranscriptionError("Set stt_base_url and stt_model for the openai-compatible provider")
        return OpenAICompatTranscriber(
            name="openai-compatible",
            base_url=settings.stt_base_url,
            model=settings.stt_model,
            api_key=env.compatible_api_key,
        )

    raise TranscriptionError(f"Unknown transcription provider: {provider}")
