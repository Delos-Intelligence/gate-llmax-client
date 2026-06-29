"""``client.total_usage`` accumulates each call's cost, mirroring the in-process end-of-run total.

Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, RawUsage
from gate_llmax.models.request import LLMRequest
from gate_llmax.models.response import LLMResponse, StreamChunk
from gate_llmax.types import OutputStatus


async def _stream_costing(self: LLMClient, request: LLMRequest, *, priority: int = 0):  # noqa: ARG001
    """Stub for ``LLMClient._stream`` whose terminal frame carries server-computed cost."""
    yield StreamChunk(text="hi")
    yield StreamChunk(is_done=True, input_tokens=10, output_tokens=20, input_cost=0.01, output_cost=0.02, provider="p")


def _send_costing(input_cost: float, output_cost: float) -> Any:
    """Stub for ``LLMClient._send`` that reports a fixed per-call cost."""

    async def _send(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
        return LLMResponse(
            raw_text="hi",
            model="m",
            status=OutputStatus.SUCCESS,
            usage=RawUsage(model="m", input_cost=input_cost, output_cost=output_cost),
        )

    return _send


def test_total_usage_accumulates_and_resets(monkeypatch: Any) -> None:
    """Every call adds its cost; ``reset_total_usage()`` starts a fresh window."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_costing(0.01, 0.02))
    client = LLMClient(api_key="k", base_url="http://x")
    assert client.total_usage == 0.0

    asyncio.run(client.request(prompt="a", operation="test_total_usage").call("m"))
    asyncio.run(client.request(prompt="b", operation="test_total_usage").call("m"))
    assert client.total_usage == pytest.approx(0.06)  # 2 × (0.01 + 0.02)

    client.reset_total_usage()
    assert client.total_usage == 0.0


def test_total_usage_excludes_disable_usage(monkeypatch: Any) -> None:
    """A ``disable_usage=True`` call is not counted toward ``total_usage``."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_costing(0.01, 0.02))
    client = LLMClient(api_key="k", base_url="http://x")

    asyncio.run(client.request(prompt="a", operation="test_total_usage").call("m"))
    asyncio.run(client.request(prompt="b", operation="test_total_usage").call("m", disable_usage=True))
    assert client.total_usage == pytest.approx(0.03)  # only the billed call counted


def test_streaming_surfaces_cost_to_usage_callback(monkeypatch: Any) -> None:
    """The streaming terminal frame's server-computed cost reaches usage_callback + total_usage."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream_costing)
    seen: list[RawUsage] = []

    async def usage_cb(usage: RawUsage) -> None:
        seen.append(usage)

    client = LLMClient(api_key="k", base_url="http://x", usage_callback=usage_cb)

    async def drain() -> None:
        async for _ in client.request(prompt="a", operation="test_total_usage").call_stream("m"):
            pass

    asyncio.run(drain())
    assert len(seen) == 1
    assert seen[0].input_cost == 0.01
    assert seen[0].output_cost == 0.02
    assert client.total_usage == pytest.approx(0.03)


async def _stream_two_usage_frames(self: LLMClient, request: LLMRequest, *, priority: int = 0):  # noqa: ARG001
    """Stub yielding two usage-bearing frames (to distinguish per-chunk vs end firing)."""
    yield StreamChunk(text="hi")
    yield StreamChunk(input_tokens=5, output_tokens=5, input_cost=0.01, output_cost=0.0, provider="p")
    yield StreamChunk(is_done=True, input_tokens=10, output_tokens=10, input_cost=0.02, output_cost=0.0, provider="p")


def test_usage_chunks_controls_callback_frequency(monkeypatch: Any) -> None:
    """Default fires once at the end; usage_chunks=True fires on every usage-bearing frame."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream_two_usage_frames)

    def run(*, usage_chunks: bool) -> int:
        seen: list[RawUsage] = []

        async def cb(usage: RawUsage) -> None:
            seen.append(usage)

        client = LLMClient(api_key="k", base_url="http://x", usage_callback=cb)

        async def drain() -> None:
            async for _ in client.request(prompt="a", operation="test_total_usage").call_stream("m", usage_chunks=usage_chunks):
                pass

        asyncio.run(drain())
        return len(seen)

    assert run(usage_chunks=False) == 1  # one DB write at the end
    assert run(usage_chunks=True) == 2  # one per usage-bearing frame
