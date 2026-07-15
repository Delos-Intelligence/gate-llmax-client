"""Per-call usage opt-out via ``disable_usage=True``.

``.call(disable_usage=True)`` / ``.call_stream(disable_usage=True)`` skip the usage callbacks
for a single call, leaving the billing callback registered for every other call.

Self-contained — stubs ``LLMClient._send`` / ``LLMClient._stream`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, RawUsage
from gate_llmax.models.request import LLMRequest
from gate_llmax.models.response import LLMResponse, StreamChunk
from gate_llmax.types import OutputStatus


async def _send_ok(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
    return LLMResponse(raw_text="hi", model="m", status=OutputStatus.SUCCESS, usage=RawUsage(model="m"))


async def _stream_ok(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> AsyncIterator[StreamChunk]:  # noqa: ARG001
    yield StreamChunk(text="hi")
    yield StreamChunk(input_tokens=1, output_tokens=2, api_provider="p")


def test_disable_usage_skips_billing_but_keeps_it_registered(monkeypatch: Any) -> None:
    """A billing-enabled client bills a normal call but skips the one passed ``disable_usage=True``."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    billed: list[RawUsage] = []

    async def usage_cb(usage: RawUsage) -> None:
        billed.append(usage)

    client = LLMClient(api_key="k", base_url="http://x", usage_callback=usage_cb)

    asyncio.run(client.request(prompt="a", operation="test_disable_usage").call("m"))
    assert len(billed) == 1  # billed normally

    asyncio.run(client.request(prompt="b", operation="test_disable_usage").call("m", disable_usage=True))
    assert len(billed) == 1  # the unbilled call did NOT fire the callback

    asyncio.run(client.request(prompt="c", operation="test_disable_usage").call("m"))
    assert len(billed) == 2  # callback still registered for later calls


def test_disable_usage_skips_per_request_on_usage(monkeypatch: Any) -> None:
    """``disable_usage=True`` also skips the per-request ``on_usage`` hook."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    seen: list[RawUsage] = []

    async def on_usage(usage: RawUsage) -> None:
        seen.append(usage)

    client = LLMClient(api_key="k", base_url="http://x")
    asyncio.run(client.request(prompt="a", on_usage=on_usage, operation="test_disable_usage").call("m", disable_usage=True))
    assert seen == []


def test_call_stream_disable_usage_skips_billing(monkeypatch: Any) -> None:
    """Streaming honours ``disable_usage``: the final-chunk usage is not billed."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream_ok)
    billed: list[RawUsage] = []

    async def usage_cb(usage: RawUsage) -> None:
        billed.append(usage)

    client = LLMClient(api_key="k", base_url="http://x", usage_callback=usage_cb)

    async def drain(*, disable_usage: bool) -> None:
        async for _ in client.request(prompt="a", operation="test_disable_usage").call_stream("m", disable_usage=disable_usage):
            pass

    asyncio.run(drain(disable_usage=False))
    assert len(billed) == 1  # billed normally

    asyncio.run(drain(disable_usage=True))
    assert len(billed) == 1  # unbilled stream did NOT fire the callback
