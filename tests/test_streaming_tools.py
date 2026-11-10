"""Streaming tool executor: live progress surfaces, ToolResult feeds back, ``redo`` ends the loop.

Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, ToolProgress, ToolResult, ToolStreamItem
from gate_llmax.models.messages import ImageMessage, TextMessage
from gate_llmax.models.response import StreamChunk
from gate_llmax.types import JsonDict

_TOOL_DELTA: list[JsonDict] = [{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": '{"q": "x"}'}}]


def _scripted_stream() -> Any:
    """Stub for ``LLMClient._stream``: the first turn calls a tool, later turns answer."""
    state = {"i": 0}

    async def _stream(self: LLMClient, request: Any, *, priority: int = 0) -> AsyncIterator[StreamChunk]:  # noqa: ARG001
        turn = state["i"]
        state["i"] += 1
        if turn == 0:
            yield StreamChunk(text="thinking... ")
            yield StreamChunk(tool_calls_delta=_TOOL_DELTA)
        else:
            yield StreamChunk(text="final answer")

    return _stream


async def _collect(builder: Any, model: str = "m") -> str:
    return "".join([chunk.text async for chunk in builder.call_stream(model) if chunk.text])


def test_progress_surfaces_and_loop_retriggers(monkeypatch: Any) -> None:
    """ToolProgress is streamed live; a ``redo=True`` result re-invokes the model for a final answer."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _scripted_stream())

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        assert name == "search"
        assert args == {"q": "x"}
        yield ToolProgress(content="[searching] ")
        yield ToolResult(output="found 3", redo=True)

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    assert asyncio.run(_collect(builder)) == "thinking... [searching] final answer"


def test_redo_false_ends_the_loop(monkeypatch: Any) -> None:
    """A ``redo=False`` result ends the loop after the tool — no second model turn."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _scripted_stream())

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        yield ToolProgress(content="[done] ")
        yield ToolResult(output="answer", redo=False)

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    assert asyncio.run(_collect(builder)) == "thinking... [done] "


def test_images_reach_the_model_inside_the_tool_message(monkeypatch: Any) -> None:
    """A result's images go into the same tool message as its text, for the next model turn to see."""
    sent: list[Any] = []
    scripted = _scripted_stream()

    async def _stream(self: LLMClient, request: Any, *, priority: int = 0) -> AsyncIterator[StreamChunk]:
        sent.append([message.model_copy(deep=True) for message in request.messages])
        async for chunk in scripted(self, request, priority=priority):
            yield chunk

    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream)

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        yield ToolResult(output="[image returned]", images=[ImageMessage(b64="iVBORw0KGgo=", media_type="image/png")])

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    asyncio.run(_collect(builder))

    tool_message = sent[1][-1]
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.content == [
        TextMessage(text="[image returned]"),
        ImageMessage(b64="iVBORw0KGgo=", media_type="image/png"),
    ]


def test_added_tools_are_offered_from_the_next_turn(monkeypatch: Any) -> None:
    """A result's named ``add_tools`` reach the next turn's tool list, once each, beside the ones already offered."""
    offered: list[list[str]] = []
    scripted = _scripted_stream()

    async def _stream(self: LLMClient, request: Any, *, priority: int = 0) -> AsyncIterator[StreamChunk]:
        offered.append([tool["function"]["name"] for tool in request.tools or []])
        async for chunk in scripted(self, request, priority=priority):
            yield chunk

    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream)
    contacts = {"type": "function", "function": {"name": "get_contacts"}}
    search = {"type": "function", "function": {"name": "search"}}

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        yield ToolResult(output="loaded", add_tools=[contacts, contacts, search, {"type": "function"}])

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    asyncio.run(_collect(builder))

    assert offered == [["search"], ["search", "get_contacts"]]
    assert [tool["function"]["name"] for tool in builder.tools] == ["search"]


def _stream_calling(arguments: str) -> Any:
    """Stub for ``LLMClient._stream``: the first turn calls ``search`` with ``arguments``, later turns answer."""
    sent: list[Any] = []
    state = {"i": 0}

    async def _stream(self: LLMClient, request: Any, *, priority: int = 0) -> AsyncIterator[StreamChunk]:  # noqa: ARG001
        sent.append([message.model_copy(deep=True) for message in request.messages])
        turn = state["i"]
        state["i"] += 1
        if turn == 0:
            yield StreamChunk(tool_calls_delta=[{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": arguments}}])
        else:
            yield StreamChunk(text="final answer")

    _stream.sent = sent  # type: ignore[attr-defined]
    return _stream


def test_non_object_arguments_reach_the_model_not_the_executor(monkeypatch: Any) -> None:
    """Valid JSON that is not an object is answered with an error the model reads; the executor never sees it."""
    stream = _stream_calling('[{"q": "x"}]')
    monkeypatch.setattr(client_mod.LLMClient, "_stream", stream)
    called: list[Any] = []

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        called.append(args)
        yield ToolResult(output="never")

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    assert asyncio.run(asyncio.wait_for(_collect(builder), timeout=5)) == "final answer"
    assert called == []
    tool_message = stream.sent[1][-1]
    assert tool_message.tool_call_id == "call_1"
    assert "must be a JSON object" in str(tool_message.content)


def test_a_raising_executor_surfaces_instead_of_hanging(monkeypatch: Any) -> None:
    """An executor that raises ends the stream with its exception; the loop no longer waits for a done marker forever."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _stream_calling('{"q": "x"}'))

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        raise AttributeError("boom")
        yield ToolResult(output="never")  # pragma: no cover

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    with pytest.raises(AttributeError, match="boom"):
        asyncio.run(asyncio.wait_for(_collect(builder), timeout=5))
