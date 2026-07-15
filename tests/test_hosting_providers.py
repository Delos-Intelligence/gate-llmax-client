"""``hosting_providers`` propagation: client default → request, per-call ``.hosting(...)`` override.

Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, RawUsage
from gate_llmax.models.request import LLMRequest
from gate_llmax.models.response import LLMResponse
from gate_llmax.types import OutputStatus

sent: list[LLMRequest] = []


async def _send_capture(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
    sent.append(request)
    return LLMResponse(raw_text="hi", model="m", status=OutputStatus.SUCCESS, usage=RawUsage(model="m"))


def _call(client: LLMClient, *hosting: str, widen: bool = False) -> LLMRequest:
    sent.clear()
    builder = client.request(prompt="a", operation="test_hosting")
    if hosting or widen:
        builder = builder.hosting(*hosting)
    asyncio.run(builder.call("m"))
    return sent[0]


def test_no_default_no_call_sends_none(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_capture)
    client = LLMClient(api_key="k", base_url="http://x")
    assert _call(client).hosting_providers is None


def test_client_default_flows_into_every_request(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_capture)
    client = LLMClient(api_key="k", base_url="http://x", default_hosting_providers=["azure", "aws-bedrock"])
    assert _call(client).hosting_providers == ["azure", "aws-bedrock"]


def test_per_call_hosting_overrides_default(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_capture)
    client = LLMClient(api_key="k", base_url="http://x", default_hosting_providers=["azure"])
    assert _call(client, "scaleway", "grok").hosting_providers == ["scaleway", "grok"]


def test_hosting_without_args_widens_back(monkeypatch: Any) -> None:
    """``.hosting()`` clears the client default for this call (parity with ``.zone(ZoneSelection())``)."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_capture)
    client = LLMClient(api_key="k", base_url="http://x", default_hosting_providers=["azure"])
    assert _call(client, widen=True).hosting_providers is None
