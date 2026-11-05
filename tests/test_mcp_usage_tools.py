"""The MCP usage tools pass the new filters through and show the new fields."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from gate_llmax.agent import mcp_server
from gate_llmax.client import LLMClient

BASE_URL = "https://gate.test"


def serve(monkeypatch: pytest.MonkeyPatch, body: Any) -> list[httpx.URL]:
    """Point the tools at a fake gateway answering ``body``; returns the URLs it was asked."""
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=body)

    def client() -> LLMClient:
        http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
        return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http)

    monkeypatch.setattr(mcp_server, "_client", client)
    return seen


@pytest.mark.asyncio
async def test_usage_latency_tool_filters_by_key_and_shows_ttft_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool sends `key` and shows both coverage counts."""
    seen = serve(monkeypatch, [{"model": "gpt-5.6", "calls": 10, "ttft_calls": 7, "stream_calls": 8}])
    [row] = await mcp_server.usage_latency(since="24h", keys=["cosmos prod"])
    assert httpx.QueryParams(seen[0].query.decode()).get_list("key") == ["cosmos prod"]
    assert (row["ttft_calls"], row["stream_calls"]) == (7, 8)


@pytest.mark.asyncio
async def test_usage_timeseries_tool_shows_the_duration_p50(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each bucket shows its median duration."""
    serve(monkeypatch, [{"t": "2026-11-04T10:00:00Z", "calls": 3, "dur_p50": 1800}])
    [point] = await mcp_server.usage_timeseries(since="2h")
    assert point["dur_p50"] == 1800


@pytest.mark.asyncio
async def test_usage_stats_tool_shows_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row shows its cached input tokens."""
    serve(monkeypatch, [{"name": "gpt-5.6", "calls": 3, "input_tokens": 1000, "cached_input_tokens": 640}])
    [row] = await mcp_server.usage_stats(since="24h")
    assert row["cached_tok"] == 640
