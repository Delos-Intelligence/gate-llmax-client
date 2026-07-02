"""Map Anthropic-on-Bedrock streaming events into shared OpenAI-shaped chunk fields.

``StreamChunk.from_bedrock_event`` handles text deltas AND ``tool_use`` content blocks,
so a Bedrock-served model streams tool calls identically to an OpenAI one. Pure parsing
test — no gateway.
"""

from __future__ import annotations

import json
from typing import Any

from gate_llmax.models.response import StreamChunk


def _accumulate(events: list[dict[str, Any]]) -> tuple[str, dict[int, dict[str, Any]]]:
    """Fold the stream the way a consumer does: text concatenated, tool deltas keyed by index."""
    text = ""
    tools: dict[int, dict[str, Any]] = {}
    for event in events:
        chunk = StreamChunk.from_bedrock_event(event)
        text += chunk.text
        for delta in chunk.tool_calls_delta or []:
            idx = delta.get("index", 0)
            call = tools.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            if delta.get("id"):
                call["id"] = delta["id"]
            fn = delta.get("function") or {}
            if fn.get("name"):
                call["name"] = fn["name"]
            call["arguments"] += fn.get("arguments") or ""
    return text, tools


def test_bedrock_tool_use_stream_reconstructs_the_call() -> None:
    """A text block followed by a tool_use block: text streams, the tool call reassembles."""
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_A", "name": "get_weather"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'ty": "Paris"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 38}},
        {"type": "message_stop"},
    ]
    text, tools = _accumulate(events)
    assert text == "Let me check."
    assert list(tools) == [1]
    assert tools[1]["id"] == "toolu_A"
    assert tools[1]["name"] == "get_weather"
    assert json.loads(tools[1]["arguments"]) == {"city": "Paris"}


def test_bedrock_parallel_tool_calls_keep_distinct_indices() -> None:
    """Two tool_use blocks accumulate independently by content-block index."""
    events: list[dict[str, Any]] = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t0", "name": "a"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "b"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'}},
    ]
    _text, tools = _accumulate(events)
    assert tools[0] == {"id": "t0", "name": "a", "arguments": "{}"}
    assert tools[1] == {"id": "t1", "name": "b", "arguments": '{"x": 1}'}


def test_bedrock_plain_text_emits_no_tool_deltas() -> None:
    """A text-only stream carries no ``tool_calls_delta`` (regression guard)."""
    events: list[dict[str, Any]] = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
    ]
    text, tools = _accumulate(events)
    assert text == "hello world"
    assert tools == {}
