"""Streaming tool executor: live progress surfaces, ToolResult feeds back, ``redo`` ends the loop.

Self-contained — stubs ``LLMClient._send`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

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
