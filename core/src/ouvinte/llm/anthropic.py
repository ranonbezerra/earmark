"""Anthropic Messages API client."""

from __future__ import annotations

import httpx

from .base import DEFAULT_TIMEOUT, LLMError, LLMProvider, describe_http_error

DEFAULT_MODEL = "claude-sonnet-5-5"
API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://api.anthropic.com",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model or DEFAULT_MODEL
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key, "anthropic-version": API_VERSION},
            timeout=DEFAULT_TIMEOUT,
            transport=transport,
        )

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = await self._client.post("/v1/messages", json=body)
            response.raise_for_status()
            blocks = response.json()["content"]
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except httpx.HTTPError as exc:
            raise LLMError(describe_http_error(self.name, exc)) from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LLMError("anthropic returned an unexpected response") from exc
        if not text.strip():
            raise LLMError("anthropic returned an empty reply")
        return text

    async def aclose(self) -> None:
        await self._client.aclose()
