"""``call_best(greatest=/lowest=)`` routes server-side to a ``BestTarget`` and returns the winner.

Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import gate_llmax.client as client_mod
from gate_llmax import LLMClient
from gate_llmax.models.request import BestTarget, LLMRequest
from gate_llmax.models.response import LLMResponse, RawUsage
from gate_llmax.types import OutputStatus


def _send_capturing() -> tuple[Any, dict[str, LLMRequest]]:
    captured: dict[str, LLMRequest] = {}

    async def _send(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
        captured["req"] = request
        return LLMResponse(raw_text="best", model="winner", status=OutputStatus.SUCCESS, usage=RawUsage(model="winner"))

    return _send, captured


def test_call_best_greatest_builds_best_target(monkeypatch: Any) -> None:
    send, captured = _send_capturing()
    monkeypatch.setattr(client_mod.LLMClient, "_send", send)
    client = LLMClient(api_key="k", base_url="http://x")

    resp = asyncio.run(client.request(prompt="hi", operation="test_best_target").call_best(["a", "b"], greatest="field_weight"))
    assert resp.raw_text == "best"
    target = captured["req"].target
    assert isinstance(target, BestTarget)
    assert target.models == ["a", "b"]
    assert target.attribute == "field_weight"
    assert target.direction == "greatest"


def test_call_best_lowest_direction(monkeypatch: Any) -> None:
    send, captured = _send_capturing()
    monkeypatch.setattr(client_mod.LLMClient, "_send", send)
    client = LLMClient(api_key="k", base_url="http://x")

    asyncio.run(client.request(prompt="hi", operation="test_best_target").call_best(["a", "b"], lowest="price"))
    target = captured["req"].target
    assert isinstance(target, BestTarget)
    assert target.attribute == "price"
    assert target.direction == "lowest"


def test_call_best_rejects_both() -> None:
    client = LLMClient(api_key="k", base_url="http://x")
    with pytest.raises(ValueError, match="not both"):
        asyncio.run(client.request(prompt="hi", operation="test_best_target").call_best(["a"], greatest="w", lowest="w"))
