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


def test_a_raising_executor_surfaces_its_error(monkeypatch: Any) -> None:
    """An executor that raises ends the stream with its error, instead of leaving the loop waiting for it forever."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _scripted_stream())

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        raise RuntimeError("tool blew up")
        yield  # an async generator, like every streaming executor

    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )
    with pytest.raises(RuntimeError, match="tool blew up"):
        asyncio.run(asyncio.wait_for(_collect(builder), timeout=2))


def _builder_with_a_slow_tool(events: list[str]) -> Any:
    """A tool that reports progress, then works for a while; ``events`` records whether it was cancelled or ran on."""

    async def executor(tool_id: str, name: str, args: dict) -> AsyncIterator[ToolStreamItem]:  # noqa: ARG001
        yield ToolProgress(content="[searching] ")
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            events.append("tool cancelled")
            raise
        events.append("tool ran on")
        yield ToolResult(output="found 3", redo=True)

    client = LLMClient(api_key="k", base_url="http://x")
    return client.request(prompt="hi", operation="test_streaming_tools").with_tools(
        [{"type": "function", "function": {"name": "search"}}], stream_executor=executor
    )


def test_a_caller_closing_mid_tool_cancels_the_tool(monkeypatch: Any) -> None:
    """A caller who closes the stream on a tool's progress cancels that tool, instead of leaving it running unseen."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _scripted_stream())
    events: list[str] = []
    builder = _builder_with_a_slow_tool(events)

    async def leave_on_progress() -> None:
        stream = builder.call_stream("m")
        async for chunk in stream:
            if chunk.text == "[searching] ":
                break
        await stream.aclose()
        events.append("caller closed")
        await asyncio.sleep(0.3)

    asyncio.run(leave_on_progress())
    assert events == ["tool cancelled", "caller closed"]


def test_a_cancelled_caller_cancels_the_running_tool(monkeypatch: Any) -> None:
    """Cancelling the task reading the stream while a tool runs cancels that tool too."""
    monkeypatch.setattr(client_mod.LLMClient, "_stream", _scripted_stream())
    events: list[str] = []
    builder = _builder_with_a_slow_tool(events)

    async def stop_on_progress() -> None:
        progressed = asyncio.Event()

        async def consume() -> None:
            async for chunk in builder.call_stream("m"):
                if chunk.text == "[searching] ":
                    progressed.set()

        consumer = asyncio.create_task(consume())
        await progressed.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        events.append("caller cancelled")
        await asyncio.sleep(0.3)

    asyncio.run(stop_on_progress())
    assert events == ["tool cancelled", "caller cancelled"]
