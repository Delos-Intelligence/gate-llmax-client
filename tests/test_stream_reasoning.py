"""``StreamChunk.from_openai`` captures the separate ``reasoning`` field (DeepSeek/GLM via OpenRouter)."""

from __future__ import annotations

from types import SimpleNamespace

from gate_llmax.models.response import StreamChunk


def _chunk(*, content: str | None = None, reasoning: str | None = None, reasoning_content: str | None = None) -> SimpleNamespace:
    """Minimal stand-in for an OpenAI ChatCompletionChunk with one choice/delta."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        delta.reasoning = reasoning
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)


def test_from_openai_captures_reasoning() -> None:
    sc = StreamChunk.from_openai(_chunk(reasoning="thinking..."))
    assert sc.reasoning == "thinking..."
    assert sc.text == ""


def test_from_openai_captures_reasoning_content_alias() -> None:
    sc = StreamChunk.from_openai(_chunk(reasoning_content="pondering"))
    assert sc.reasoning == "pondering"


def test_from_openai_no_reasoning_is_empty() -> None:
    sc = StreamChunk.from_openai(_chunk(content="hello"))
    assert sc.reasoning == ""
    assert sc.text == "hello"
