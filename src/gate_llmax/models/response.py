"""Response models for the Gate LLM Gateway."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..types import JsonDict, OutputStatus
from .vision import VisionOCR


class ToolFunction(BaseModel):
    """The function being called by a tool call (OpenAI-shaped)."""

    model_config = ConfigDict(extra="allow")
    name: str = ""
    arguments: str = ""


class ToolCall(BaseModel):
    """One tool call requested by the assistant (OpenAI-shaped)."""

    model_config = ConfigDict(extra="allow")
    id: str = ""
    type: Literal["function"] = "function"
    function: ToolFunction = Field(default_factory=ToolFunction)

    @classmethod
    def from_openai(cls, tool_call: Any) -> ToolCall:
        """Build from an OpenAI-shaped tool call (a dict or any object with the same fields)."""
        return cls.model_validate(tool_call, from_attributes=True)


# Cost decimals. Matches the scale of Gate's usage_logs.cost / usage_rollup.cost
# columns (numeric(_, 10)) — anything finer is truncated by the database anyway.
# Rounding shallower than this silently drops whole calls: at 5 decimals every
# request costing under $0.000005 (a short gpt-4.1-nano or gpt-4o-mini call)
# recorded as exactly 0.
ROUND = 10


class RawUsage(BaseModel):
    """Token usage and cost for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0  # subset of output_tokens spent on the reasoning chain; informational, not billed twice
    input_cost: float = 0.0
    output_cost: float = 0.0
    model: str = ""
    estimated: bool = False
    api_provider: str = ""
    hosting_provider: str = ""
    region: str = ""
    duration_ms: int = 0
    ttft_ms: int | None = None
    operation: str = ""
    finish_reason: str = ""  # why generation stopped (stop/length/tool_calls/…); "length" on an empty answer means the cap was hit

    @computed_field
    @property
    def total_cost(self) -> float:
        """Total cost of the call."""
        return round(self.input_cost + self.output_cost, ROUND)

    @computed_field
    @property
    def provider(self) -> str:
        """Deprecated alias of ``api_provider`` — kept on the wire for pre-v0.3 clients."""
        return self.api_provider

    def __add__(self, other: RawUsage) -> RawUsage:
        """Combine two usage objects."""
        return RawUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            input_cost=round(self.input_cost + other.input_cost, ROUND),
            output_cost=round(self.output_cost + other.output_cost, ROUND),
            model=self.model or other.model,
            estimated=self.estimated or other.estimated,
            api_provider=self.api_provider or other.api_provider,
            hosting_provider=self.hosting_provider or other.hosting_provider,
            region=self.region or other.region,
            duration_ms=self.duration_ms + other.duration_ms,
            ttft_ms=self.ttft_ms if self.ttft_ms is not None else other.ttft_ms,
            operation=self.operation or other.operation,
            finish_reason=self.finish_reason or other.finish_reason,
        )


class StreamChunkDelta(BaseModel):
    """Delta content for a streaming chunk (OpenAI compat)."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[JsonDict] | None = None


class StreamChunkChoice(BaseModel):
    """Choice for a streaming chunk (OpenAI compat)."""

    delta: StreamChunkDelta
    finish_reason: str | None = None


class StreamChunkUsage(BaseModel):
    """Usage for a streaming chunk (OpenAI compat)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


class StreamChunk(BaseModel):
    """Streaming chunk; `.choices` / `.usage` match OpenAI-style events."""

    text: str = ""
    reasoning: str = ""
    is_done: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    tool_calls_delta: list[JsonDict] | None = None
    api_provider: str | None = None
    hosting_provider: str | None = None
    region: str | None = None
    duration_ms: int | None = None
    ttft_ms: int | None = None

    @computed_field
    @property
    def provider(self) -> str | None:
        """Deprecated alias of ``api_provider`` — kept on the wire for pre-v0.3 clients."""
        return self.api_provider

    @property
    def choices(self) -> list[StreamChunkChoice]:
        """OpenAI-compatible choices list."""
        return [
            StreamChunkChoice(
                delta=StreamChunkDelta(
                    content=self.text or None,
                    reasoning_content=self.reasoning or None,
                    tool_calls=self.tool_calls_delta,
                ),
                finish_reason=self.finish_reason,
            ),
        ]

    @property
    def usage(self) -> StreamChunkUsage | None:
        """OpenAI-compatible usage, present only on the final chunk."""
        if self.input_tokens is not None or self.output_tokens is not None:
            return StreamChunkUsage(
                prompt_tokens=self.input_tokens or 0,
                completion_tokens=self.output_tokens or 0,
            )
        return None

    def to_openai_chunk(self, *, model: str, completion_id: str, created: int, with_usage: bool = False) -> JsonDict:
        """Serialize as an OpenAI ChatCompletionChunk dict.

        ``with_usage`` mirrors ``stream_options.include_usage`` — the final chunk (which
        carries token counts) then includes an OpenAI-shaped ``usage`` block.
        """
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": c.delta.model_dump(exclude_none=True), "finish_reason": c.finish_reason} for c in self.choices
            ],
        }
        if with_usage and self.usage is not None:
            payload["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
        return payload

    @classmethod
    def from_openai(cls, chunk: Any) -> StreamChunk:
        """Build from an OpenAI ChatCompletionChunk."""
        text = ""
        reasoning = ""
        is_done = False
        finish_reason = None
        input_tokens = None
        output_tokens = None
        tool_calls_delta = None

        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta:
                if choice.delta.content:
                    text = choice.delta.content
                # Reasoning streamed in a separate field (DeepSeek/GLM via OpenRouter).
                reasoning = getattr(choice.delta, "reasoning", None) or getattr(choice.delta, "reasoning_content", None) or ""
                if choice.delta.tool_calls:
                    # Plain dicts so StreamChunk stays JSON-serializable over SSE.
                    tool_calls_delta = [tc.model_dump() for tc in choice.delta.tool_calls]
            if choice.finish_reason:
                is_done = True
                finish_reason = choice.finish_reason

        cached_input_tokens = None
        if chunk.usage:
            input_tokens = getattr(chunk.usage, "prompt_tokens", None)
            output_tokens = getattr(chunk.usage, "completion_tokens", None)
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            if details is not None:
                cached_input_tokens = getattr(details, "cached_tokens", None)

        return cls(
            text=text,
            reasoning=reasoning,
            is_done=is_done,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            tool_calls_delta=tool_calls_delta,
        )

    @classmethod
    def from_bedrock_event(cls, event: dict[str, Any]) -> StreamChunk:
        """Build from an AWS Bedrock streaming event."""
        event_type = event.get("type", "")
        text = ""
        is_done = False
        finish_reason = None
        input_tokens = None
        output_tokens = None
        cached_input_tokens = None
        tool_calls_delta = None

        if event_type == "content_block_start":
            # A tool_use block opens with its id + name; the argument JSON streams in
            # the following input_json_delta events under the same content-block index.
            # Emit an OpenAI-shaped opening delta so downstream accumulators (keyed by
            # index) see the call — text blocks carry no content_block.type and are skipped.
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                tool_calls_delta = [
                    {
                        "index": event.get("index", 0),
                        "id": block.get("id"),
                        "type": "function",
                        "function": {"name": block.get("name", ""), "arguments": ""},
                    }
                ]
        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "input_json_delta":
                # One tool-argument JSON fragment; index matches the opening block.
                tool_calls_delta = [
                    {
                        "index": event.get("index", 0),
                        "function": {"arguments": delta.get("partial_json", "")},
                    }
                ]
            else:
                text = delta.get("text", "")
        elif event_type == "message_start":
            message = event.get("message", {})
            usage = message.get("usage", {})
            # Anthropic reports input_tokens EXCLUDING cached/created; add them back for the true prompt total.
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
            base_input = usage.get("input_tokens", 0) or 0
            input_tokens = base_input + cache_read + cache_creation
            cached_input_tokens = cache_read
        elif event_type == "message_delta":
            usage = event.get("usage", {})
            output_tokens = usage.get("output_tokens")
            delta = event.get("delta", {})
            finish_reason = delta.get("stop_reason")
        elif event_type == "message_stop":
            is_done = True

        return cls(
            text=text,
            is_done=is_done,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            tool_calls_delta=tool_calls_delta,
        )


class LLMCallRecord(BaseModel):
    """Metadata shared by every Gate call (chat, images, audio, tts, embed).

    Covers *what* was done (``model``, ``deployment_id``), *how it went* (``status``,
    ``latency_ms``), and *what it cost* (``usage``). Payload-specific fields live on
    the per-endpoint response models (e.g. ``LLMResponse.raw_text``, ``ImageResponse.data``).
    """

    usage: RawUsage = Field(default_factory=RawUsage)
    model: str = ""
    deployment_id: UUID | None = None
    status: OutputStatus = OutputStatus.SUCCESS
    detail: str = Field(
        default="",
        description=(
            "Why a non-SUCCESS status happened, in the provider's own words (truncated). Empty on "
            "success. The status says a request was refused; this says what to change about it."
        ),
    )
    latency_ms: int = 0
    cached: bool = Field(default=False, description="True when this response was replayed from the gateway response cache.")


class BaseAudioResponse(LLMCallRecord):
    """Shared base for audio-producing responses (TTS speech + generative music/sfx/dialogue)."""

    audio: str = Field(default="", description="Base64-encoded generated audio bytes.")


class LLMResponse(LLMCallRecord):
    """Response from a chat call — call metadata plus the assistant payload.

    Three orthogonal payload slots:

    - ``raw_text``: assistant text reply (empty string when none).
    - ``tool_calls``: list of OpenAI-shaped tool call dicts (``None`` when no tools fired).
    - ``json_object``: structured dict — either set server-side (e.g. vision OCR)
      or filled by ``RequestBuilder.cast_json()`` after the call. ``None`` by default.

    For Pydantic-typed parsing, use ``RequestBuilder.cast(T)`` — it reads from
    ``json_object`` first, then falls back to extracting JSON from ``raw_text``.
    """

    raw_text: str = ""
    reasoning: str = Field(default="", description="Assistant reasoning/thinking text (empty string when none).")
    tool_calls: list[ToolCall] | None = None
    json_object: JsonDict | None = None
    choices: list[str] | None = Field(
        default=None,
        description="All completion texts when `specifics.n` > 1 (`raw_text` is `choices[0]`); None for a single completion.",
    )

    @classmethod
    def no_deployment(cls, model: str) -> LLMResponse:
        """Synthetic response for "model unknown or no active deployment"."""
        return cls(usage=RawUsage(model=model), model=model, status=OutputStatus.NO_DEPLOYMENT)

    @classmethod
    def timeout(cls, model: str) -> LLMResponse:
        """Synthetic response for a per-model call that exceeded the batch deadline."""
        return cls(usage=RawUsage(model=model, estimated=True), model=model, status=OutputStatus.TIMEOUT)

    def to_openai_completion(self, *, model: str, completion_id: str, created: int) -> JsonDict:
        """Serialize as an OpenAI ChatCompletion dict (single choice; ``finish_reason`` from tool calls)."""
        message: dict[str, Any] = {"role": "assistant", "content": self.raw_text or None}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if self.tool_calls else "stop",
                    "message": message,
                },
            ],
            "usage": {
                "prompt_tokens": self.usage.input_tokens,
                "completion_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
            },
        }


class VisionLLMResponse(LLMCallRecord):
    """Response from a vision OCR call — call metadata plus the typed OCR result."""

    vision: VisionOCR | None = None


class MulticallStreamFrame(BaseModel):
    """One SSE frame from a streaming parallel multicall.

    Emitted in completion order (not request order); use ``index`` to map back
    to the position in the original ``ParallelTarget.models`` list.
    """

    index: int
    model: str
    response: LLMResponse
