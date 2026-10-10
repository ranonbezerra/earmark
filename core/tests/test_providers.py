import asyncio
import json

import httpx
import pytest
from conftest import make_wav

from ouvinte.config import Env, UserSettings
from ouvinte.llm import AnthropicProvider, LLMError, OpenAICompatProvider, build_llm
from ouvinte.transcription import (
    MlxWhisperTranscriber,
    OpenAICompatTranscriber,
    TranscriptionError,
    build_transcriber,
)


def chat_reply(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_local_server_uses_first_listed_model_and_classic_parameters():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen-mlx"}, {"id": "other"}]})
        return httpx.Response(200, json=chat_reply("hello"))

    provider = OpenAICompatProvider(
        name="omlx", base_url="http://localhost:8000/v1", api_key="k", transport=httpx.MockTransport(handler)
    )

    assert asyncio.run(provider.complete(system="s", prompt="p", max_tokens=99)) == "hello"
    body = json.loads(seen[-1].content)
    assert seen[-1].url.path == "/v1/chat/completions"
    assert seen[-1].headers["authorization"] == "Bearer k"
    assert body["model"] == "qwen-mlx"
    assert body["max_tokens"] == 99 and "temperature" in body
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]


def test_openai_uses_modern_parameters():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_reply("ok"))

    provider = OpenAICompatProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="some-model",
        modern_parameters=True,
        transport=httpx.MockTransport(handler),
    )
    asyncio.run(provider.complete(system="s", prompt="p", max_tokens=50))

    assert seen[0]["max_completion_tokens"] == 50
    assert "max_tokens" not in seen[0] and "temperature" not in seen[0]


def test_connection_and_http_errors_become_readable_llm_errors():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    offline = OpenAICompatProvider(
        name="omlx", base_url="http://localhost:8000/v1", model="m", transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(LLMError, match="could not connect to omlx at http://localhost:8000"):
        asyncio.run(offline.complete(system="s", prompt="p"))

    failing = OpenAICompatProvider(
        name="omlx",
        base_url="http://localhost:8000/v1",
        model="m",
        transport=httpx.MockTransport(lambda _r: httpx.Response(500, text="model crashed")),
    )
    with pytest.raises(LLMError, match="HTTP 500: model crashed"):
        asyncio.run(failing.complete(system="s", prompt="p"))


def test_server_without_models_is_explained():
    provider = OpenAICompatProvider(
        name="omlx",
        base_url="http://localhost:8000/v1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"data": []})),
    )
    with pytest.raises(LLMError, match="no models available"):
        asyncio.run(provider.complete(system="s", prompt="p"))


def test_anthropic_request_shape_and_reply():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "part one, "}, {"type": "text", "text": "part two"}]},
        )

    provider = AnthropicProvider(api_key="sk-test", transport=httpx.MockTransport(handler))

    assert asyncio.run(provider.complete(system="sys", prompt="p", max_tokens=10)) == "part one, part two"
    request = seen[0]
    body = json.loads(request.content)
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "sys" and body["max_tokens"] == 10
    assert body["messages"] == [{"role": "user", "content": "p"}]


def test_llm_factory_follows_settings(tmp_path):
    env = Env(data_dir=tmp_path, anthropic_api_key="a", openai_api_key="o")

    omlx = build_llm(UserSettings(), env)
    assert isinstance(omlx, OpenAICompatProvider) and omlx.name == "omlx" and omlx.model is None

    claude = build_llm(UserSettings(llm_provider="anthropic"), env)
    assert isinstance(claude, AnthropicProvider) and claude.model.startswith("claude-")

    openai = build_llm(UserSettings(llm_provider="openai", llm_model="m"), env)
    assert isinstance(openai, OpenAICompatProvider) and openai.name == "openai"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (UserSettings(llm_provider="anthropic"), "ANTHROPIC_API_KEY"),
        (UserSettings(llm_provider="openai", llm_model="m"), "OPENAI_API_KEY"),
        (UserSettings(llm_provider="openai-compatible"), "llm_base_url"),
    ],
)
def test_llm_factory_explains_missing_configuration(tmp_path, settings, message):
    with pytest.raises(LLMError, match=message):
        build_llm(settings, Env(data_dir=tmp_path))


def test_api_transcription_sends_the_file_and_reads_segments(tmp_path):
    audio = make_wav(tmp_path / "system-00000.wav", 1.0)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "text": "olá mundo",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": " olá"},
                    {"start": 0.4, "end": 0.9, "text": " mundo"},
                    {"start": 0.9, "end": 1.0, "text": "  "},
                ],
            },
        )

    transcriber = OpenAICompatTranscriber(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="whisper-1",
        api_key="o",
        transport=httpx.MockTransport(handler),
    )
    chunks = transcriber.transcribe(audio, language="pt")

    assert [(c.start_s, c.end_s, c.text) for c in chunks] == [(0.0, 0.4, "olá"), (0.4, 0.9, "mundo")]
    body = seen[0].content
    assert seen[0].url.path == "/v1/audio/transcriptions"
    assert b'name="model"' in body and b"whisper-1" in body
    assert b'name="language"' in body and b'filename="system-00000.wav"' in body


def test_api_transcription_without_segments_covers_the_whole_file(tmp_path):
    audio = make_wav(tmp_path / "mic-00000.wav", 2.0)
    transcriber = OpenAICompatTranscriber(
        name="x",
        base_url="http://stt.local/v1",
        model="m",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"text": "tudo junto"})),
    )

    assert [(c.start_s, c.end_s, c.text) for c in transcriber.transcribe(audio)] == [(0.0, 2.0, "tudo junto")]


def test_api_transcription_errors_are_readable(tmp_path):
    audio = make_wav(tmp_path / "mic-00000.wav", 0.2)
    transcriber = OpenAICompatTranscriber(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="whisper-1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(401, text="bad key")),
    )

    with pytest.raises(TranscriptionError, match="HTTP 401 for mic-00000.wav: bad key"):
        transcriber.transcribe(audio)


def test_transcriber_factory_follows_settings(tmp_path):
    env = Env(data_dir=tmp_path, openai_api_key="o")

    assert isinstance(build_transcriber(UserSettings(), env), MlxWhisperTranscriber)
    remote = build_transcriber(UserSettings(stt_provider="openai"), env)
    assert isinstance(remote, OpenAICompatTranscriber) and remote.model == "whisper-1"
    with pytest.raises(TranscriptionError, match="OPENAI_API_KEY"):
        build_transcriber(UserSettings(stt_provider="openai"), Env(data_dir=tmp_path))
    with pytest.raises(TranscriptionError, match="stt_base_url"):
        build_transcriber(UserSettings(stt_provider="openai-compatible"), env)
