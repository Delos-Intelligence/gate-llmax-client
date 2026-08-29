"""Duplicate tool names must never reach the provider (Bedrock 400 'Tool names must be unique')."""

from __future__ import annotations

from typing import Any

from gate_llmax.models.request import LLMRequest, SingleTarget


def _tool(name: str, description: str = "") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description}}


def _request(tools: list[dict[str, Any]]) -> LLMRequest:
    return LLMRequest(target=SingleTarget(model="claude-5-opus"), tools=tools)


def _names(request: LLMRequest) -> list[str]:
    assert request.tools is not None
    return [t["function"]["name"] for t in request.tools]


def test_duplicates_are_dropped_keeping_first() -> None:
    request = _request([_tool("read", "first"), _tool("write"), _tool("read", "second")])
    assert _names(request) == ["read", "write"]
    assert request.tools is not None
    assert request.tools[0]["function"]["description"] == "first"


def test_no_duplicates_is_unchanged() -> None:
    request = _request([_tool("read"), _tool("write")])
    assert _names(request) == ["read", "write"]


def test_flat_name_shape_is_deduped() -> None:
    request = LLMRequest(target=SingleTarget(model="m"), tools=[{"name": "read"}, {"name": "read"}, {"name": "list"}])
    assert request.tools is not None
    assert [t["name"] for t in request.tools] == ["read", "list"]


def test_unnamed_entries_are_kept() -> None:
    request = LLMRequest(target=SingleTarget(model="m"), tools=[{"type": "function"}, _tool("read"), _tool("read")])
    assert request.tools is not None
    assert len(request.tools) == 2


def test_none_and_empty_are_preserved() -> None:
    assert LLMRequest(target=SingleTarget(model="m")).tools is None
    assert LLMRequest(target=SingleTarget(model="m"), tools=[]).tools == []
