"""``LLMClient.simple_request`` — a terse builder factory over ``.request(...)``.

Returns a normal builder, so the shared ``.call``/``.call_prefer`` terminals run it.
Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, TextMessage
from gate_llmax.models.request import LLMRequest, SingleTarget
from gate_llmax.models.response import LLMResponse, RawUsage
from gate_llmax.types import OutputStatus


def _send_returning(text: str) -> tuple[Any, dict[str, LLMRequest]]:
    """Stub for ``LLMClient._send`` plus the dict it records the sent request into."""
    captured: dict[str, LLMRequest] = {}

    async def _send(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
        captured["req"] = request
        return LLMResponse(raw_text=text, model="m", status=OutputStatus.SUCCESS, usage=RawUsage(model="m"))

    return _send, captured


def test_flat_tuning_and_shared_call(monkeypatch: Any) -> None:
    """Flat kwargs become ``specifics``; the shared ``.call(model)`` runs it and returns the response."""
    send, captured = _send_returning("hello")
    monkeypatch.setattr(client_mod.LLMClient, "_send", send)
    client = LLMClient(api_key="k", base_url="http://x")

    resp = asyncio.run(client.simple_request("hi", operation="op", temperature=0.2, max_tokens=64).call("m"))
    assert resp.raw_text == "hello"
    req = captured["req"]
    assert req.specifics.temperature == 0.2
    assert req.specifics.max_tokens == 64
    assert req.operation == "op"
    assert isinstance(req.target, SingleTarget)
    assert req.target.model == "m"
    block = req.messages[0].content[0]
    assert isinstance(block, TextMessage)
    assert block.text == "hi"


def test_cast_json_chains_to_parsed_dict(monkeypatch: Any) -> None:
    """For JSON, ``.cast_json()`` chains off the builder: forces JSON mode AND returns the parsed dict."""
    send, captured = _send_returning('{"a": 1}')
    monkeypatch.setattr(client_mod.LLMClient, "_send", send)
    client = LLMClient(api_key="k", base_url="http://x")

    resp = asyncio.run(client.simple_request("hi", temperature=0).cast_json().call("m"))
    assert resp.json_response == {"a": 1}  # parsed — no manual json.loads on raw_text
    assert captured["req"].response_format == {"type": "json_object"}  # server-side JSON forced


def test_chains_with_call_prefer_and_fluent(monkeypatch: Any) -> None:
    """The returned builder is normal: ``.zone(...)`` chains and ``.call_prefer([...])`` falls back in order."""
    send, captured = _send_returning("hello")
    monkeypatch.setattr(client_mod.LLMClient, "_send", send)
    client = LLMClient(api_key="k", base_url="http://x")

    resp = asyncio.run(client.simple_request("hi", temperature=0).zone("EU").call_prefer(["a", "b"]))
    assert resp.raw_text == "hello"
    req = captured["req"]
    assert isinstance(req.target, SingleTarget)
    assert req.target.model == "a"
    assert req.zone_selection is not None
    assert req.zone_selection.regions == ["EU"]
