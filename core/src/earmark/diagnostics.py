"""Checks whether the configured pieces are actually usable right now."""

from __future__ import annotations

from .llm import LLMError, OpenAICompatProvider
from .service import AppContext
from .transcription import MlxWhisperTranscriber, TranscriptionError
from .transcription import mlx_whisper as mlx_whisper_backend


async def check_providers(ctx: AppContext) -> dict:
    settings = ctx.settings_store.load()
    report: dict = {
        "capture": {
            "available": ctx.recorder.available,
            "detail": None if ctx.recorder.available else "helper not built; run `make capture`",
        }
    }

    stt: dict = {"provider": settings.stt_provider, "model": settings.stt_model, "available": False}
    try:
        transcriber = ctx.transcriber_factory(settings, ctx.env)
        stt["model"] = transcriber.model
        if isinstance(transcriber, MlxWhisperTranscriber) and not mlx_whisper_backend.is_available():
            stt["detail"] = "mlx-whisper is not installed; run `uv sync --extra local`"
        else:
            stt["available"] = True
        transcriber.close()
    except TranscriptionError as exc:
        stt["detail"] = str(exc)
    report["stt"] = stt

    llm: dict = {"provider": settings.llm_provider, "model": settings.llm_model, "available": False}
    try:
        provider = ctx.llm_factory(settings, ctx.env)
        try:
            if isinstance(provider, OpenAICompatProvider):
                # Listing models doubles as a reachability check.
                llm["models"] = await provider.list_models()
                llm["model"] = settings.llm_model or (llm["models"][0] if llm["models"] else None)
                if llm["model"] is None:
                    llm["detail"] = "the server is up but has no models"
                else:
                    llm["available"] = True
            else:
                llm["model"] = provider.model
                llm["available"] = True
        finally:
            await provider.aclose()
    except LLMError as exc:
        llm["detail"] = str(exc)
    report["llm"] = llm

    wanted = [
        name
        for name, on in (("notes", settings.export_notes), ("reminders", settings.export_reminders))
        if on
    ]
    export: dict = {"targets": wanted, "available": True}
    if wanted and ctx.exporter_factory(settings) is None:
        export["available"] = False
        export["detail"] = "Notes and Reminders can only be reached on macOS"
    report["export"] = export
    return report
