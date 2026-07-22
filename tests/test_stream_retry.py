"""A dropped SSE connection retries before the first chunk and raises typed after it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from gate_llmax import LLMClient
from gate_llmax.exceptions import LLMConnectionError


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


def make_client(handler) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url="http://gate.test", transport=transport)
    return LLMClient(api_key="k", base_url="http://gate.test", httpx_aclient=http)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep the retry timing out of the test's wall clock."""
    monkeypatch.setattr("gate_llmax.client.STREAM_RETRY_BACKOFF", 0.0)


async def test_a_drop_before_the_first_chunk_is_retried():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            msg = "connection dropped"
            raise httpx.ReadError(msg)
        body = sse({"text": "pong"}) + sse({"text": "", "is_done": True, "finish_reason": "stop"}) + DONE
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = make_client(handler)
    chunks = [c async for c in client.request(prompt="hi", operation="test").call_stream("m")]

    assert attempts == 3
    assert "".join(c.text for c in chunks) == "pong"


async def test_retries_are_bounded():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        msg = "connection dropped"
        raise httpx.ReadError(msg)

    client = make_client(handler)
    with pytest.raises(LLMConnectionError, match="failed to start"):
        _ = [c async for c in client.request(prompt="hi", operation="test").call_stream("m")]

    assert attempts == 3  # the first try plus STREAM_MAX_RETRIES


async def test_a_drop_after_the_first_chunk_raises_instead_of_resending():
    attempts = 0
    seen: list[str] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=DroppingStream([sse({"text": "par"}), sse({"text": "tial"})]),
        )

    client = make_client(handler)

    async def drain() -> None:
        async for chunk in client.request(prompt="hi", operation="test").call_stream("m"):
            seen.append(chunk.text)  # noqa: PERF401 - the point is what arrives before the raise

    with pytest.raises(LLMConnectionError, match="interrupted"):
        await drain()

    assert seen == ["par", "tial"]  # what arrived is kept, not replayed
    assert attempts == 1
