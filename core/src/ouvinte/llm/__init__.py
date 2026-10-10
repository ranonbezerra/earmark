"""Summary model providers and the factory that picks one from settings."""

from __future__ import annotations

from ..config import Env, UserSettings
from .anthropic import AnthropicProvider
from .base import LLMError, LLMProvider
from .openai_compat import OpenAICompatProvider

OMLX_BASE_URL = "http://localhost:8000/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

__all__ = [
    "AnthropicProvider",
    "LLMError",
    "LLMProvider",
    "OpenAICompatProvider",
    "build_llm",
]


def build_llm(settings: UserSettings, env: Env) -> LLMProvider:
    provider = settings.llm_provider

    if provider == "omlx":
        # Runs on this machine. With no model set, the first one oMLX lists is used.
        return OpenAICompatProvider(
            name="omlx",
            base_url=settings.llm_base_url or OMLX_BASE_URL,
            model=settings.llm_model,
            api_key=env.omlx_api_key,
        )

    if provider == "openai":
        if not env.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        if not settings.llm_model:
            raise LLMError("Set llm_model to the OpenAI model you want to use")
        return OpenAICompatProvider(
            name="openai",
            base_url=settings.llm_base_url or OPENAI_BASE_URL,
            model=settings.llm_model,
            api_key=env.openai_api_key,
            modern_parameters=True,
        )

    if provider == "anthropic":
        if not env.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        kwargs = {"base_url": settings.llm_base_url} if settings.llm_base_url else {}
        return AnthropicProvider(api_key=env.anthropic_api_key, model=settings.llm_model, **kwargs)

    if provider == "openai-compatible":
        if not settings.llm_base_url:
            raise LLMError("Set llm_base_url for the openai-compatible provider")
        return OpenAICompatProvider(
            name="openai-compatible",
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=env.compatible_api_key,
        )

    raise LLMError(f"Unknown LLM provider: {provider}")
