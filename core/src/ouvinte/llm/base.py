"""Common interface for the models that write summaries."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

# Local models can take minutes on a long block of transcript.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)


class LLMError(RuntimeError):
    """The provider could not be reached or returned something unusable."""


class LLMProvider(ABC):
    name: str
    model: str | None

    @abstractmethod
    async def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        """Returns the model's text reply to ``prompt``."""

    async def resolved_model(self) -> str | None:
        """The model actually used (some providers pick one when none is set)."""
        return self.model

    async def aclose(self) -> None:  # noqa: B027  (optional hook)
        pass


def describe_http_error(provider: str, exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300].strip()
        return f"{provider} answered HTTP {exc.response.status_code}: {body}"
    if isinstance(exc, httpx.ConnectError):
        url = exc.request.url
        port = f":{url.port}" if url.port else ""
        return f"could not connect to {provider} at {url.scheme}://{url.host}{port}"
    if isinstance(exc, httpx.TimeoutException):
        return f"{provider} timed out"
    return f"{provider} request failed: {exc}"
