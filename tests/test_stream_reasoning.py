"""``StreamChunk.from_openai`` captures the separate ``reasoning`` field (DeepSeek/GLM via OpenRouter)."""

from __future__ import annotations

from types import SimpleNamespace

from gate_llmax.models.response import LLMResponse, StreamChunk


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


def test_to_openai_chunk_carries_reasoning() -> None:
    """One spelling goes out, whichever one came in."""
    chunk = StreamChunk(reasoning="abc").to_openai_chunk(model="m", completion_id="c", created=0)
    assert chunk["choices"][0]["delta"]["reasoning"] == "abc"
    assert "reasoning_content" not in chunk["choices"][0]["delta"]


def test_to_openai_chunk_omits_reasoning_when_text_only() -> None:
    chunk = StreamChunk(text="hi").to_openai_chunk(model="m", completion_id="c", created=0)
    assert "reasoning" not in chunk["choices"][0]["delta"]
    assert chunk["choices"][0]["delta"]["content"] == "hi"


def test_to_openai_completion_carries_reasoning() -> None:
    completion = LLMResponse(reasoning="xyz", raw_text="hi").to_openai_completion(model="m", completion_id="c", created=0)
    message = completion["choices"][0]["message"]
    assert message["reasoning"] == "xyz"
    assert message["content"] == "hi"


def test_to_openai_completion_omits_reasoning_when_empty() -> None:
    completion = LLMResponse(raw_text="hi").to_openai_completion(model="m", completion_id="c", created=0)
    assert "reasoning" not in completion["choices"][0]["message"]
