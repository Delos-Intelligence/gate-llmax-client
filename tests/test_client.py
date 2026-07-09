"""Integration tests for LLMClient against a running Gate server.

Self-contained — no dependency on the backend source tree. Config comes from environment
variables, optionally loaded from a ``.env`` placed beside this file (see ``.env.example``);
real environment variables take precedence. Local-dev defaults:
  GATE_BASE_URL      - Gate server URL              (default: http://localhost:8000)
  GATE_API_KEY       - consumer API key             (default: gate-local-default-key)
  GATE_CHAT_MODEL    - a deployed chat model name   (default: gemini-flash-latest)
  GATE_EMBED_MODEL / GATE_IMAGES_MODEL / GATE_TTS_MODEL / GATE_STT_MODEL / GATE_AUDIO_GEN_MODEL
                     - deployed model names; the matching tests skip when unset
  GATE_VISION_MODEL  - a deployed vision model      (default: azure_vision)
  GATE_TTS_VOICE     - provider-specific voice id
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from gate_llmax import (
    BaseAudioResponse,
    ImageResponse,
    LLMAuthError,
    LLMBudgetError,
    LLMClient,
    LLMResponse,
    ModelInfo,
    ModelPurpose,
    RawUsage,
    VisionLLMResponse,
    VisionOCR,
)
from gate_llmax.types import OutputStatus

TESTS_DIR = Path(__file__).resolve().parent
load_dotenv(TESTS_DIR / ".env")

BASE_URL = os.getenv("GATE_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("GATE_API_KEY", "gate-local-default-key")
CHAT_MODEL = os.getenv("GATE_CHAT_MODEL", "gemini-flash-latest")
EMBED_MODEL = os.getenv("GATE_EMBED_MODEL", "")
IMAGES_MODEL = os.getenv("GATE_IMAGES_MODEL", "")
TTS_MODEL = os.getenv("GATE_TTS_MODEL", "")
TTS_VOICE = os.getenv("GATE_TTS_VOICE", "21m00Tcm4TlvDq8ikWAM")  # provider-specific voice id
AUDIO_GEN_MODEL = os.getenv("GATE_AUDIO_GEN_MODEL", "")
STT_MODEL = os.getenv("GATE_STT_MODEL", "")  # transcription model (e.g. whisper-1)
VISION_MODEL = os.getenv("GATE_VISION_MODEL", "azure_vision")
POEM_PNG = TESTS_DIR / "poem.png"


@pytest.fixture
async def client() -> AsyncGenerator[LLMClient, None]:
    async with LLMClient(api_key=API_KEY, base_url=BASE_URL) as c:
        yield c


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


async def test_list_models(client: LLMClient) -> None:
    models = await client.list_models()
    assert isinstance(models, list)
    assert all(isinstance(m, ModelInfo) for m in models)
    assert all(isinstance(m.purpose, ModelPurpose) for m in models)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def test_call(client: LLMClient) -> None:
    resp = await client.request(operation="test_chat", prompt="Reply with just the word OK.").call(CHAT_MODEL)
    assert isinstance(resp, LLMResponse)
    assert resp.status == OutputStatus.SUCCESS
    assert isinstance(resp.raw_text, str)
    assert resp.raw_text
    assert resp.usage.output_tokens > 0


async def test_call_stream(client: LLMClient) -> None:
    parts: list[str] = []
    async for chunk in client.request(operation="test_chat", prompt="Count 1 to 3 separated by commas, no other text.").call_stream(
        CHAT_MODEL
    ):
        if chunk.text:
            parts.append(chunk.text)
        if chunk.is_done:
            break
    assert len(parts) > 0
    assert "1" in "".join(parts)


async def test_stream_smooth_client_paces(client: LLMClient) -> None:
    """Client-side smoothing sleeps ``smooth_duration_ms`` after each text chunk.

    We tie the wall-clock floor to the *observed* number of text chunks, so the check is
    immune to model non-determinism about chunk count while still proving pacing happened.
    """
    duration_ms = 50
    n_text_chunks = 0
    parts: list[str] = []
    start = time.monotonic()
    async for chunk in (
        client.request(operation="test_smooth", prompt="Count from 1 to 20, separated by single spaces, no other text.")
        .stream(smooth=True, smooth_duration_ms=duration_ms)
        .call(CHAT_MODEL)
    ):
        if chunk.text:
            parts.append(chunk.text)
            n_text_chunks += 1
        if chunk.is_done:
            break
    elapsed_ms = (time.monotonic() - start) * 1000
    assert "".join(parts).strip()
    assert n_text_chunks >= 3, f"expected a multi-chunk stream, got {n_text_chunks}"
    assert elapsed_ms >= n_text_chunks * duration_ms * 0.5, f"{elapsed_ms:.0f}ms too fast for {n_text_chunks} paced chunks"


async def test_stream_smooth_server_side(client: LLMClient) -> None:
    """``server_side=True`` moves pacing onto the gateway (it sleeps per text chunk).

    Same wall-clock floor as the client path — the gateway's delay is included in the time
    the client waits between chunks — so this proves the gateway honoured ``smooth``.
    """
    duration_ms = 40
    n_text_chunks = 0
    parts: list[str] = []
    start = time.monotonic()
    async for chunk in (
        client.request(operation="test_smooth", prompt="Count from 1 to 20, separated by single spaces, no other text.")
        .stream(smooth=True, server_side=True, smooth_duration_ms=duration_ms)
        .call(CHAT_MODEL)
    ):
        if chunk.text:
            parts.append(chunk.text)
            n_text_chunks += 1
        if chunk.is_done:
            break
    elapsed_ms = (time.monotonic() - start) * 1000
    assert "".join(parts).strip()
    assert n_text_chunks >= 3, f"expected a multi-chunk stream, got {n_text_chunks}"
    assert elapsed_ms >= n_text_chunks * duration_ms * 0.5, f"{elapsed_ms:.0f}ms too fast for {n_text_chunks} paced chunks"


async def test_multicall(client: LLMClient) -> None:
    responses = await client.request(operation="test_chat", prompt="Reply with just the word OK.").multicall([CHAT_MODEL, CHAT_MODEL])
    assert len(responses) == 2
    assert all(isinstance(r, LLMResponse) for r in responses)
    assert all(r.status == OutputStatus.SUCCESS for r in responses)


# ---------------------------------------------------------------------------
# Cast (typed response)
# ---------------------------------------------------------------------------


class CityFact(BaseModel):
    city: str
    one_fact: str


async def test_cast(client: LLMClient) -> None:
    typed = await (
        client.request(
            system_prompt='Reply with a single JSON object only, no markdown: {"city": string, "one_fact": string}.',
            prompt="Amsterdam",
            operation="test_chat",
        )
        .cast(CityFact)
        .call(CHAT_MODEL)
    )
    assert typed.status == OutputStatus.SUCCESS
    assert isinstance(typed.value, CityFact)
    assert isinstance(typed.value.city, str)
    assert isinstance(typed.value.one_fact, str)


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


async def test_embed(client: LLMClient) -> None:
    if not EMBED_MODEL:
        pytest.skip("GATE_EMBED_MODEL not set")
    resp = await client.embed("Hello world", operation="test_embed").call(EMBED_MODEL)
    assert len(resp.data) > 0
    assert all(isinstance(v, float) for v in resp.data[0].embedding)


# ---------------------------------------------------------------------------
# Images (generate + edit, non-streaming and streaming)
# ---------------------------------------------------------------------------


async def test_images_generate(client: LLMClient) -> None:
    if not IMAGES_MODEL:
        pytest.skip("GATE_IMAGES_MODEL not set")
    seen_usages: list[RawUsage] = []
    seen_responses: list[ImageResponse] = []

    async def counter(u: RawUsage) -> None:
        seen_usages.append(u)

    async def on_response(r: ImageResponse) -> None:
        seen_responses.append(r)

    resp = await (
        client.image("a red square on white background", operation="test_images", size=(1024, 1024))
        .callback(on_response)
        .usage_callback(counter)
        .call(IMAGES_MODEL)
    )
    assert resp.data
    assert len(resp.data[0].b64) > 100
    assert resp.usage.input_tokens >= 1
    assert len(seen_usages) == 1
    assert seen_usages[0] is resp.usage
    assert len(seen_responses) == 1


async def test_images_stream(client: LLMClient) -> None:
    if not IMAGES_MODEL:
        pytest.skip("GATE_IMAGES_MODEL not set")
    seen_usages: list[RawUsage] = []

    async def counter(u: RawUsage) -> None:
        seen_usages.append(u)

    frames = [
        f
        async for f in (
            client.image("a green circle", operation="test_images", size=(1024, 1024), partial_images=2)
            .usage_callback(counter)
            .call_stream(IMAGES_MODEL)
        )
    ]
    assert len(frames) >= 1
    assert all(len(f.b64) > 100 for f in frames)
    assert len(seen_usages) == 1  # terminal usage frame fires once


# ---------------------------------------------------------------------------
# TTS (non-streaming and streaming)
# ---------------------------------------------------------------------------


async def test_tts_call(client: LLMClient) -> None:
    if not TTS_MODEL:
        pytest.skip("GATE_TTS_MODEL not set")
    text = "Hello world from Gate."
    seen_usages: list[RawUsage] = []
    seen_responses: list[BaseAudioResponse] = []

    async def counter(u: RawUsage) -> None:
        seen_usages.append(u)

    async def on_response(r: BaseAudioResponse) -> None:
        seen_responses.append(r)

    resp = (
        await client.audio("speech", operation="test_tts", text=text, voice=TTS_VOICE)
        .callback(on_response)
        .usage_callback(counter)
        .call(TTS_MODEL)
    )
    assert len(resp.audio) > 100
    assert resp.usage.input_tokens == len(text)
    assert len(seen_usages) == 1
    assert len(seen_responses) == 1


async def test_tts_stream(client: LLMClient) -> None:
    if not TTS_MODEL:
        pytest.skip("GATE_TTS_MODEL not set")
    text = "Streaming hello from Gate."
    chunks = [c async for c in client.audio("speech", operation="test_tts", text=text, voice=TTS_VOICE).call_stream(TTS_MODEL)]
    assert sum(len(c) for c in chunks) > 100


async def test_tts_then_stt_roundtrip(client: LLMClient) -> None:
    """Full audio round-trip: synthesize a known phrase (TTS), then transcribe it back (STT).

    Proves both media directions over the gate agree — the transcript recovers the spoken words.
    """
    if not TTS_MODEL:
        pytest.skip("GATE_TTS_MODEL not set")
    if not STT_MODEL:
        pytest.skip("GATE_STT_MODEL not set")

    phrase = "The quick brown fox jumps over the lazy dog."
    speech = await client.audio("speech", operation="test_roundtrip_tts", text=phrase, voice=TTS_VOICE).call(TTS_MODEL)
    assert len(speech.audio) > 100  # base64 mp3 bytes

    transcript = await client.transcribe(speech.audio, operation="test_roundtrip_stt", language="en").call(STT_MODEL)
    text = transcript.text.lower()
    assert text.strip(), "empty transcript"
    # ASR can wobble on function words; the content words should survive the round-trip.
    assert "fox" in text
    assert "dog" in text


# ---------------------------------------------------------------------------
# Response caching (per-call cache_ttl)
# ---------------------------------------------------------------------------


async def test_cache_roundtrip(client: LLMClient) -> None:
    """A second identical call with cache_ttl replays the first; without it, nothing is cached.

    Skips when the gateway has no cache configured (no REDIS_URL).
    """
    prompt = f"Reply with the single word KIWI. token={uuid.uuid4()}"
    first = await client.request(operation="test_chat", prompt=prompt, cache_ttl=120).call(CHAT_MODEL)
    second = await client.request(operation="test_chat", prompt=prompt, cache_ttl=120).call(CHAT_MODEL)
    if not second.cached:
        pytest.skip("gateway response cache not enabled (no REDIS_URL)")
    assert first.cached is False
    assert second.raw_text == first.raw_text

    plain_prompt = f"Reply with the single word KIWI. token={uuid.uuid4()}"
    a = await client.request(operation="test_chat", prompt=plain_prompt).call(CHAT_MODEL)
    b = await client.request(operation="test_chat", prompt=plain_prompt).call(CHAT_MODEL)
    assert a.cached is False
    assert b.cached is False


async def test_cache_client_default() -> None:
    """A client-level default cache_ttl caches by default; a per-call 0 forces it off."""
    cached_client = LLMClient(api_key=API_KEY, base_url=BASE_URL, cache_ttl=120)
    try:
        prompt = f"Reply with the single word PEAR. token={uuid.uuid4()}"
        await cached_client.request(operation="test_chat", prompt=prompt).call(CHAT_MODEL)
        second = await cached_client.request(operation="test_chat", prompt=prompt).call(CHAT_MODEL)
        if not second.cached:
            pytest.skip("gateway response cache not enabled (no REDIS_URL)")
        # Per-call cache_ttl=0 forces caching off even though the client default is on.
        off_prompt = f"Reply with the single word PEAR. token={uuid.uuid4()}"
        await cached_client.request(operation="test_chat", prompt=off_prompt, cache_ttl=0).call(CHAT_MODEL)
        again = await cached_client.request(operation="test_chat", prompt=off_prompt, cache_ttl=0).call(CHAT_MODEL)
        assert again.cached is False
    finally:
        await cached_client.close()


# ---------------------------------------------------------------------------
# Generative audio (sound effects)
# ---------------------------------------------------------------------------


async def test_generate_audio_sound_effects(client: LLMClient) -> None:
    if not AUDIO_GEN_MODEL:
        pytest.skip("GATE_AUDIO_GEN_MODEL not set")
    prompt = "a short metallic click"
    resp = await client.audio("sound_effects", operation="test_audio_gen", prompt=prompt, duration_seconds=2.0).call(AUDIO_GEN_MODEL)
    assert resp.model == AUDIO_GEN_MODEL  # Gate model name, not the upstream id
    assert len(base64.b64decode(resp.audio)) > 100
    assert resp.usage.input_tokens == len(prompt)


# ---------------------------------------------------------------------------
# Universal usage callback (chat + images + tts share the same counter)
# ---------------------------------------------------------------------------


async def test_usage_callback_uniform(client: LLMClient) -> None:
    """A single `UsageCallback` registered via `.usage_callback(...)` must fire from chat, images, and tts."""
    seen: list[RawUsage] = []

    async def counter(u: RawUsage) -> None:
        seen.append(u)

    # Chat
    await client.request(operation="test_chat", prompt="Reply with just OK.").usage_callback(counter).call(CHAT_MODEL)

    # Images (skip if not configured)
    if IMAGES_MODEL:
        await client.image("a tiny dot", operation="test_images", size=(1024, 1024)).usage_callback(counter).call(IMAGES_MODEL)

    # TTS (skip if not configured)
    if TTS_MODEL:
        await client.audio("speech", operation="test_tts", text="hi", voice=TTS_VOICE).usage_callback(counter).call(TTS_MODEL)

    expected = 1 + (1 if IMAGES_MODEL else 0) + (1 if TTS_MODEL else 0)
    assert len(seen) == expected
    assert all(isinstance(u, RawUsage) for u in seen)


# ---------------------------------------------------------------------------
# Fallback / batch / tools / vision
# ---------------------------------------------------------------------------


async def test_call_prefer_falls_back(client: LLMClient) -> None:
    """call_prefer skips an unknown model and returns the next working one."""
    resp = await client.request(operation="test_chat", prompt="Reply with just OK.").call_prefer(
        ["definitely-not-a-real-model-xyz", CHAT_MODEL]
    )
    assert resp.status == OutputStatus.SUCCESS
    # CHAT_MODEL may be a rolling alias (e.g. *-latest); the gate reports the concrete model it
    # resolved to, so require a real answer from a real model rather than an exact-name echo.
    assert resp.model
    assert resp.model != "definitely-not-a-real-model-xyz"
    assert resp.raw_text


async def test_batch_multicall(client: LLMClient) -> None:
    """batch_multicall returns one SUCCESS response per model, in order."""
    responses = await client.request(operation="test_chat", prompt="Reply with just OK.").batch_multicall([CHAT_MODEL, CHAT_MODEL])
    assert len(responses) == 2
    assert all(isinstance(r, LLMResponse) for r in responses)
    assert all(r.status == OutputStatus.SUCCESS for r in responses)


async def test_call_best_priority_skips_failed(client: LLMClient) -> None:
    """call_best (priority mode) excludes a failing model and returns the next working one."""
    resp = await client.request(operation="test_chat", prompt="Reply with just OK.").call_best(
        ["definitely-not-a-real-model-xyz", CHAT_MODEL]
    )
    assert resp.status == OutputStatus.SUCCESS
    # Alias-tolerant (see test_call_prefer_falls_back): assert a real model answered, not a name echo.
    assert resp.model
    assert resp.model != "definitely-not-a-real-model-xyz"


async def test_call_best_key(client: LLMClient) -> None:
    """call_best with a score function returns the highest-scoring SUCCESS."""
    resp = await client.request(operation="test_chat", prompt="Reply with just OK.").call_best(
        [CHAT_MODEL, CHAT_MODEL], key=lambda r: len(r.raw_text)
    )
    assert resp.status == OutputStatus.SUCCESS
    assert resp.raw_text


SECRET_COLOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_secret_color",
            "description": "Returns the secret color. Call this to learn the secret color.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def test_with_tools_executor(client: LLMClient) -> None:
    """`.with_tools(tools, executor).call()` runs the loop and feeds the result back."""
    calls: list[str] = []

    async def executor(_tool_call_id: str, name: str, _args: dict) -> str:
        calls.append(name)
        return "PURPLE"

    resp = (
        await client.request(
            system_prompt="To answer, you must call the get_secret_color tool, then reply with the color it returns.",
            prompt="What is the secret color?",
            operation="test_chat",
        )
        .with_tools(SECRET_COLOR_TOOLS, executor)
        .call(CHAT_MODEL)
    )
    assert resp.status == OutputStatus.SUCCESS
    assert "get_secret_color" in calls
    assert "purple" in resp.raw_text.lower()


async def test_with_tools_no_executor(client: LLMClient) -> None:
    """`.with_tools(tools)` with no executor records the model's requested calls in tool_calls."""
    resp = (
        await client.request(
            system_prompt="You must call the get_secret_color tool to answer.",
            prompt="What is the secret color?",
            operation="test_chat",
        )
        .with_tools(SECRET_COLOR_TOOLS)
        .call(CHAT_MODEL)
    )
    assert resp.status == OutputStatus.SUCCESS
    assert resp.tool_calls
    assert resp.tool_calls[0].function.name == "get_secret_color"


async def test_vision_ocr(client: LLMClient) -> None:
    """vision(...).call() returns a typed VisionLLMResponse with recognised text lines."""
    if not VISION_MODEL:
        pytest.skip("GATE_VISION_MODEL not set")
    image_b64 = base64.standard_b64encode(POEM_PNG.read_bytes()).decode()
    resp = await client.vision([image_b64], operation="test_vision").call(VISION_MODEL)
    assert isinstance(resp, VisionLLMResponse)
    if resp.status == OutputStatus.NO_DEPLOYMENT:
        pytest.skip("vision deployment unavailable")
    assert resp.status == OutputStatus.SUCCESS
    assert isinstance(resp.vision, VisionOCR)
    assert len(resp.vision.lines) > 0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def test_auth_error() -> None:
    async with LLMClient(api_key="invalid-key", base_url=BASE_URL) as bad_client:
        with pytest.raises(LLMAuthError):
            await bad_client.list_models()


async def test_budget_denied(client: LLMClient) -> None:
    """A .budget() check returning False denies the call before any dispatch."""

    async def deny() -> bool:
        return False

    with pytest.raises(LLMBudgetError):
        await client.request(operation="test_chat", prompt="hi").budget(deny).call(CHAT_MODEL)
