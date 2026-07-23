"""Unit tests for the dev-key usage insight client methods.

No gateway involved: a MockTransport captures the request, so these assert the part that is
actually this layer's job — how filters become query parameters and how responses are parsed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gate_llmax.client import LLMClient

BASE_URL = "https://gate.test"


def make_client(handler: Any) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url=BASE_URL, transport=transport)
    return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http)


@pytest.mark.asyncio
async def test_usage_errors_sends_repeated_filter_params() -> None:
    """Multi-valued filters go out as repeated keys, which is what the route reads."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(
            200,
            json={
                "window_from": "2026-07-23T00:00:00Z",
                "window_to": "2026-07-24T00:00:00Z",
                "keys": ["cosmos prod"],
                "total_failures": 3,
                "by_status": {"TIMEOUT": 3},
                "groups": [
                    {
                        "status": "TIMEOUT",
                        "model": "gpt-5.6",
                        "api_key": "cosmos prod",
                        "calls": 3,
                        "sample_detail": "Stream timed out",
                        "replayable": 2,
                    }
                ],
            },
        )

    async with make_client(handler) as client:
        report = await client.usage_errors(
            since="7d",
            keys=["cosmos prod"],
            statuses=["TIMEOUT", "RATE_LIMIT"],
            models=["gpt-5.6"],
        )

    params = httpx.QueryParams(seen["url"].query.decode())
    assert params["since"] == "7d"
    assert params.get_list("status") == ["TIMEOUT", "RATE_LIMIT"]
    assert params.get_list("key") == ["cosmos prod"]
    assert params.get_list("model") == ["gpt-5.6"]

    assert report.total_failures == 3
    assert report.groups[0].sample_detail == "Stream timed out"
    assert report.groups[0].replayable == 2


@pytest.mark.asyncio
async def test_usage_error_samples_omits_unset_filters() -> None:
    """An unset filter must not travel as an empty value the route would try to match."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(200, json=[])

    async with make_client(handler) as client:
        await client.usage_error_samples(since="24h")

    params = httpx.QueryParams(seen["url"].query.decode())
    assert set(params.keys()) == {"since", "limit"}


@pytest.mark.asyncio
async def test_usage_error_samples_flags_and_search() -> None:
    """`search` and `replayable_only` are the two knobs that narrow a noisy window."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(
            200,
            json=[
                {
                    "id": "1c8041f7-0000-0000-0000-000000000000",
                    "created_at": "2026-07-23T15:32:00Z",
                    "status": "NO_VALIDATION",
                    "detail": "array too long. Expected maximum length 128, but got 133",
                    "replayable": True,
                }
            ],
        )

    async with make_client(handler) as client:
        samples = await client.usage_error_samples(search="array too long", replayable_only=True)

    params = httpx.QueryParams(seen["url"].query.decode())
    assert params["search"] == "array too long"
    assert params["replayable_only"] == "true"
    assert samples[0].replayable is True
    assert "128" in samples[0].detail


@pytest.mark.asyncio
async def test_usage_payload_returns_body_and_endpoint() -> None:
    """The point of the capture: a body plus where to POST it back."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/usage/payload/abc"
        return httpx.Response(
            200,
            json={
                "id": "1c8041f7-0000-0000-0000-000000000000",
                "created_at": "2026-07-23T15:32:00Z",
                "status": "TIMEOUT",
                "request_type": "chat",
                "endpoint": "/v1/chat/completions",
                "payload": {"target": {"kind": "single", "model": "gpt-5.6"}, "messages": []},
            },
        )

    async with make_client(handler) as client:
        stored = await client.usage_payload("abc")

    assert stored.endpoint == "/v1/chat/completions"
    assert stored.payload["target"]["model"] == "gpt-5.6"
    # Round-trips as JSON, so an agent can hand it straight back to the gateway.
    assert json.loads(json.dumps(stored.payload)) == stored.payload


@pytest.mark.asyncio
async def test_list_api_key_names_never_exposes_a_secret() -> None:
    """Key names are the addressing scheme; the secret must never come back with them."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "1", "name": "cosmos prod", "is_active": True}])

    async with make_client(handler) as client:
        keys = await client.list_api_key_names()

    assert keys[0].name == "cosmos prod"
    assert not hasattr(keys[0], "key_hash")
