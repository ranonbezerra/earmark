"""Transcription through any server with the OpenAI audio API."""

from __future__ import annotations

from pathlib import Path

import httpx

from ..audio import wav_duration
from .base import Transcriber, TranscriptChunk, TranscriptionError, chunks_from_response

TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)


class OpenAICompatTranscriber(Transcriber):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.name = name
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=TIMEOUT, transport=transport
        )

    def transcribe(self, path: Path, language: str | None = None) -> list[TranscriptChunk]:
        data = {"model": self.model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        try:
            response = self._post(path, data)
            if response.status_code == 400 and "response_format" in response.text:
                # Some models only offer plain JSON, without timestamps.
                response = self._post(path, data | {"response_format": "json"})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300].strip()
            raise TranscriptionError(
                f"{self.name} answered HTTP {exc.response.status_code} for {path.name}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"{self.name} transcription request failed: {exc}") from exc
        except ValueError as exc:
            raise TranscriptionError(f"{self.name} returned an unexpected response") from exc
        if not isinstance(payload, dict):
            raise TranscriptionError(f"{self.name} returned an unexpected response")
        return chunks_from_response(payload, fallback_duration=wav_duration(path))

    def _post(self, path: Path, data: dict[str, str]) -> httpx.Response:
        with path.open("rb") as audio:
            return self._client.post(
                "/audio/transcriptions", data=data, files={"file": (path.name, audio, "audio/wav")}
            )

    def close(self) -> None:
        self._client.close()
