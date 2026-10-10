"""Chat Completions client.

One implementation covers every server that speaks the OpenAI API: oMLX,
LM Studio, Ollama and OpenAI itself.
"""

from __future__ import annotations

import httpx

from .base import DEFAULT_TIMEOUT, LLMError, LLMProvider, describe_http_error


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        # OpenAI's current models reject ``max_tokens`` and custom temperatures;
        # local servers generally expect the classic parameters.
        modern_parameters: bool = False,
        temperature: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.name = name
        self.model = model
        self._modern = modern_parameters
        self._temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=DEFAULT_TIMEOUT, transport=transport
        )

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get("/models", timeout=10.0)
            response.raise_for_status()
            return [item["id"] for item in response.json().get("data", []) if "id" in item]
        except httpx.HTTPError as exc:
            raise LLMError(describe_http_error(self.name, exc)) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise LLMError(f"{self.name} returned an unexpected model list") from exc

    async def resolved_model(self) -> str | None:
        """Falls back to the first model the server offers when none is set.

        Convenient for local servers, where "whatever is loaded" is usually
        what the user wants.
        """
        if self.model is None:
            models = await self.list_models()
            if not models:
                raise LLMError(f"{self.name} has no models available. Download or load one first.")
            self.model = models[0]
        return self.model

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        body: dict = {
            "model": await self.resolved_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        if self._modern:
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = self._temperature
        try:
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise LLMError(describe_http_error(self.name, exc)) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"{self.name} returned an empty reply")
        return content

    async def aclose(self) -> None:
        await self._client.aclose()
