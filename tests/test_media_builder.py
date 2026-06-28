"""Media builders: ``operation`` flows onto the request, and ``call_prefer`` falls back sequentially."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, LLMServerError
from gate_llmax.models.images import ImageRequest, ImageResponse
from gate_llmax.models.response import RawUsage
from gate_llmax.types import OutputStatus


def test_operation_flows_onto_media_request() -> None:
    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.image_request("a cat", operation="studio.image", timeout=30)
    assert builder.request.operation == "studio.image"
    assert builder.request.timeout == 30


def test_call_prefer_falls_back_sequentially(monkeypatch: Any) -> None:
    """call_prefer tries models in order, skipping failures, and bills only the winner."""
    calls: list[str] = []

    async def _send_images(self: LLMClient, request: ImageRequest) -> ImageResponse:  # noqa: ARG001
        calls.append(request.model)
        if request.model == "bad":
            raise LLMServerError("boom")
        return ImageResponse(model=request.model, data=[], status=OutputStatus.SUCCESS, usage=RawUsage(model=request.model))

    monkeypatch.setattr(client_mod.LLMClient, "_send_images", _send_images)
    client = LLMClient(api_key="k", base_url="http://x")

    resp = asyncio.run(client.image_request("a cat").call_prefer(["bad", "good"]))
    assert resp.model == "good"
    assert calls == ["bad", "good"]  # sequential: tried bad, then good — never fans out


def test_call_prefer_reraises_when_all_fail(monkeypatch: Any) -> None:
    async def _send_images(self: LLMClient, request: ImageRequest) -> ImageResponse:  # noqa: ARG001
        raise LLMServerError("boom")

    monkeypatch.setattr(client_mod.LLMClient, "_send_images", _send_images)
    client = LLMClient(api_key="k", base_url="http://x")

    with pytest.raises(LLMServerError):
        asyncio.run(client.image_request("a cat").call_prefer(["m1", "m2"]))
