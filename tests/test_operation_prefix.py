"""Client views: ``prefix_operation`` and ``with_usage_callback``.

Both return a copy sharing the connection pool and rate limiter, so one cached client can serve
many callers — each tagging its own operation namespace and billing its own principal — without
mutating the shared instance.

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

SENT: list[str] = []


async def _send_ok(self: LLMClient, request: LLMRequest, *, priority: int = 0) -> LLMResponse:  # noqa: ARG001
    SENT.append(request.operation)
    return LLMResponse(raw_text="hi", model="m", status=OutputStatus.SUCCESS, usage=RawUsage(model="m", operation=request.operation))


def _client() -> LLMClient:
    SENT.clear()
    return LLMClient(api_key="k", base_url="http://x")


def test_prefix_is_prepended_to_the_operation(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    asyncio.run(client.prefix_operation("scribe").request(prompt="a", operation="spellcheck").call("m"))

    assert SENT == ["scribe/spellcheck"]


def test_prefixes_accumulate_with_the_operation_last(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    scribe = client.prefix_operation("scribe")
    asyncio.run(scribe.prefix_operation("tables").request(prompt="a", operation="spellcheck").call("m"))

    assert SENT == ["scribe/tables/spellcheck"]


def test_the_original_client_is_untouched(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    client.prefix_operation("scribe")
    asyncio.run(client.request(prompt="a", operation="spellcheck").call("m"))

    assert SENT == ["spellcheck"]


def test_an_empty_prefix_changes_nothing(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    assert client.prefix_operation("") is client
    assert client.prefix_operation("/") is client


def test_a_prefix_with_no_operation_stands_alone(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    asyncio.run(client.prefix_operation("scribe").request(prompt="a", operation="").call("m"))

    assert SENT == ["scribe"]


def test_simple_request_is_prefixed_once(monkeypatch: Any) -> None:
    """``simple_request`` delegates to ``request``, so only one of the two may apply the prefix."""
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()

    asyncio.run(client.prefix_operation("scribe").simple_request("sys", operation="spellcheck").call("m"))

    assert SENT == ["scribe/spellcheck"]


def test_embed_and_media_builders_are_prefixed() -> None:
    client = _client().prefix_operation("scribe")

    assert client.embed(input="a", operation="index").request.model_dump()["operation"] == "scribe/index"
    assert client.image(prompt="a", operation="draw").request.model_dump()["operation"] == "scribe/draw"
    assert client.video(prompt="a", operation="clip").request.model_dump()["operation"] == "scribe/clip"
    assert client.vision(images=["a"], operation="ocr").request.model_dump()["operation"] == "scribe/ocr"


def test_with_usage_callback_bills_only_through_the_view(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()
    billed: list[RawUsage] = []

    async def usage_cb(usage: RawUsage) -> None:
        billed.append(usage)

    view = client.with_usage_callback(usage_cb)

    asyncio.run(view.request(prompt="a", operation="op").call("m"))
    assert len(billed) == 1

    asyncio.run(client.request(prompt="b", operation="op").call("m"))
    assert len(billed) == 1  # the shared client never got the hook


def test_with_usage_callback_keeps_the_client_defaults(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    base: list[RawUsage] = []
    extra: list[RawUsage] = []

    async def base_cb(usage: RawUsage) -> None:
        base.append(usage)

    async def extra_cb(usage: RawUsage) -> None:
        extra.append(usage)

    client = LLMClient(api_key="k", base_url="http://x", usage_callback=base_cb)
    asyncio.run(client.with_usage_callback(extra_cb).request(prompt="a", operation="op").call("m"))

    assert len(base) == 1
    assert len(extra) == 1


def test_a_view_credits_the_client_it_came_from(monkeypatch: Any) -> None:
    monkeypatch.setattr(client_mod.LLMClient, "_send", _send_ok)
    client = _client()
    view = client.prefix_operation("scribe")

    asyncio.run(view.request(prompt="a", operation="op").call("m"))

    assert client.total_usage == view.total_usage


def test_a_view_shares_the_pool_and_does_not_own_it() -> None:
    client = _client()
    view = client.prefix_operation("scribe").with_usage_callback()

    assert view._http is client._http
    assert view._limiter is client._limiter
    assert not view._owns_http


def test_closing_a_view_leaves_the_shared_pool_open() -> None:
    client = _client()
    view = client.prefix_operation("scribe")

    asyncio.run(view.close())

    assert not client._http.is_closed
