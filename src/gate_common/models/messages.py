"""Message types for Gate requests (chat roles + text / image content blocks)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import JsonDict
from .response import ToolCall


class MessageRole(StrEnum):
    """Role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextMessage(BaseModel):
    """Plain text segment inside a multimodal message."""

    model_config = ConfigDict(extra="forbid")

    text: str


class ImageMessage(BaseModel):
    """Image segment: supply exactly one of raw base64 (no data: prefix) or a URL."""

    model_config = ConfigDict(extra="forbid")

    b64: str | None = None
    url: str | None = None
    detail: str | None = Field(
        default=None,
        description='Forwarded to OpenAI-style APIs as image detail (e.g. "auto", "low", "high").',
    )

    @model_validator(mode="after")
    def exactly_one_image_source(self) -> Self:
        """Validate that exactly one of b64 or url is provided."""
        has_b64 = self.b64 is not None and self.b64 != ""
        has_url = self.url is not None and self.url != ""
        if has_b64 == has_url:
            msg = "ImageMessage requires exactly one of b64 or url (non-empty)"
            raise ValueError(msg)
        return self


class Message(BaseModel):
    """A single turn in a conversation."""

    role: MessageRole
    content: list[TextMessage | ImageMessage] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="OpenAI-shaped tool calls on an assistant turn.",
    )

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=[TextMessage(text=content)])

    @classmethod
    def assistant(cls, content: str) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=[TextMessage(text=content)])

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=[TextMessage(text=content)])

    @classmethod
    def assistant_tool_calls(cls, tool_calls: list[ToolCall], content: str = "") -> Message:
        """Create an assistant turn that requested ``tool_calls`` (optional accompanying text)."""
        blocks: list[TextMessage | ImageMessage] = [TextMessage(text=content)] if content else []
        return cls(role=MessageRole.ASSISTANT, content=blocks, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> Message:
        """Create a tool-result message answering a prior tool call."""
        return cls(role=MessageRole.TOOL, content=[TextMessage(text=content)], tool_call_id=tool_call_id, name=name)


def content_block_to_openai(block: TextMessage | ImageMessage) -> JsonDict:
    """Map a Gate content block to an OpenAI chat `content` element."""
    if isinstance(block, TextMessage):
        return {"type": "text", "text": block.text}
    detail = block.detail or "high"
    if block.url is not None:
        return {"type": "image_url", "image_url": {"url": block.url, "detail": detail}}
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{block.b64}", "detail": detail},
    }


def message_to_openai_dict(msg: Message) -> JsonDict:
    """Serialize a `Message` to an OpenAI-compatible chat message dict."""
    payload: dict[str, Any] = {"role": msg.role.value}
    if isinstance(msg.content, str):
        payload["content"] = msg.content
    elif msg.content:
        payload["content"] = [content_block_to_openai(b) for b in msg.content]
    elif msg.tool_calls:
        # Assistant tool-call turn with no text: OpenAI expects content=null.
        payload["content"] = None
    else:
        payload["content"] = []
    if msg.tool_calls is not None:
        payload["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    if msg.tool_call_id is not None:
        payload["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        payload["name"] = msg.name
    return payload


def message_plain_text(content: str | list[TextMessage | ImageMessage]) -> str:
    """Flatten message content to a single string (text segments only)."""
    if isinstance(content, str):
        return content
    parts = [block.text for block in content if isinstance(block, TextMessage)]
    return "\n".join(parts)


def last_message_plain_text(messages: list[Message]) -> str:
    """Plain-text body of the last message, for simple single-prompt APIs."""
    if not messages:
        return ""
    return message_plain_text(messages[-1].content)
