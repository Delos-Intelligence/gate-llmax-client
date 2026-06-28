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
from gate_llmax.models.response import LLMResponse
from gate_llmax.types import OutputStatus


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

    asyncio.run(client.request(prompt="a").call("m"))
    asyncio.run(client.request(prompt="b").call("m"))
    assert client.total_usage == pytest.approx(0.06)  # 2 × (0.01 + 0.02)

    client.reset_total_usage()
    assert client.total_usage == 0.0


def test_total_usage_excludes_disable_usage(monkeypatch: Any) -> None:
    """A ``disable_usage=True`` call is not counted toward ``total_usage``."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_costing(0.01, 0.02))
    client = LLMClient(api_key="k", base_url="http://x")

    asyncio.run(client.request(prompt="a").call("m"))
    asyncio.run(client.request(prompt="b").call("m", disable_usage=True))
    assert client.total_usage == pytest.approx(0.03)  # only the billed call counted
