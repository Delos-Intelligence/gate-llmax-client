"""A severed SSE connection restarts inside the client; the caller never sees the drop."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from gate_llmax import LLMClient
from gate_llmax.client import STREAM_INTERRUPTED


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class DroppingStream(httpx.AsyncByteStream):
    """Replays `chunks`, then dies the way a severed connection does."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Hold the bytes to replay before the connection dies."""
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Replay the canned bytes, then raise the transport error httpx would."""
        for chunk in self.chunks:
            yield chunk
        msg = "connection dropped"
        raise httpx.ReadError(msg)


def dropping(*chunks: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DroppingStream(list(chunks)))


def complete(*chunks: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join([*chunks, DONE]))


def make_client(handler) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="http://gate.test", transport=transport)
    return LLMClient(api_key="k", base_url="http://gate.test", httpx_aclient=http)


async def drain(client: LLMClient) -> list[str]:
    return [c.text async for c in client.request(prompt="hi", operation="test").call_stream("m")]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep the retry timing out of the test's wall clock."""
    monkeypatch.setattr("gate_llmax.client.STREAM_RETRY_BACKOFF", 0.0)


async def test_a_drop_before_the_first_chunk_replays_the_same_request():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) < 3:
            msg = "connection dropped"
            raise httpx.ReadError(msg)
        return complete(sse({"text": "pong"}), sse({"text": "", "is_done": True, "finish_reason": "stop"}))

    chunks = await drain(make_client(handler))

    assert "".join(chunks) == "pong"
    assert len(sent) == 3
    # Nothing was delivered, so nothing is fed back: every attempt is the original request.
    assert all(len(payload["messages"]) == 1 for payload in sent)


async def test_a_mid_stream_drop_resumes_from_the_text_already_delivered():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return dropping(sse({"text": "Roses are "}))
        return complete(sse({"text": "red."}), sse({"text": "", "is_done": True, "finish_reason": "stop"}))

    chunks = await drain(make_client(handler))

    # One uninterrupted sequence, no duplicated prefix — the caller cannot tell it reconnected.
    assert "".join(chunks) == "Roses are red."
    # The retry hands the model what the user already saw, as an assistant turn to continue.
    resumed = sent[1]["messages"][-1]
    assert resumed["role"] == "assistant"
    assert resumed["content"] == [{"text": "Roses are "}]
    assert len(sent[1]["messages"]) == len(sent[0]["messages"]) + 1


async def test_repeated_drops_resume_from_everything_delivered_so_far():
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return dropping(sse({"text": "one "}))
        if len(sent) == 2:
            return dropping(sse({"text": "two "}))
        return complete(sse({"text": "three"}), sse({"text": "", "is_done": True, "finish_reason": "stop"}))

    chunks = await drain(make_client(handler))

    assert "".join(chunks) == "one two three"
    # Each resume appends exactly one assistant turn to the *original* messages, never a chain.
    assert len(sent[2]["messages"]) == len(sent[0]["messages"]) + 1
    assert sent[2]["messages"][-1]["content"] == [{"text": "one two "}]


async def test_exhausted_retries_end_the_stream_instead_of_raising():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return dropping(sse({"text": "partial"}))

    chunks = [c async for c in make_client(handler).request(prompt="hi", operation="test").call_stream("m")]

    assert attempts == 3  # the first try plus STREAM_MAX_RETRIES
    assert "".join(c.text for c in chunks) == "partialpartialpartial"
    assert chunks[-1].finish_reason == STREAM_INTERRUPTED
    assert chunks[-1].is_done


async def test_a_half_streamed_tool_call_is_not_resumed():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return dropping(sse({"tool_calls_delta": [{"index": 0, "function": {"arguments": '{"q": "unf'}}]}))

    chunks = [c async for c in make_client(handler).request(prompt="hi", operation="test").call_stream("m")]

    # Re-asking would emit a second call the caller would merge into the truncated first.
    assert attempts == 1
    assert chunks[-1].finish_reason == STREAM_INTERRUPTED
