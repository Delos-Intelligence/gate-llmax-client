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

    @classmethod
    def from_openai(cls, message: Any) -> Message:
        """Build a Message from an OpenAI-shaped chat message — a dict or any object with the same fields.

        Handles system / user / assistant (incl. ``tool_calls``) / tool turns, including multimodal
        ``content`` lists: ``text`` parts become ``TextMessage`` blocks and ``image_url`` parts become
        ``ImageMessage`` blocks (data-URI base64 or remote URL). System / tool turns flatten to text.
        """
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
            tool_calls, tool_call_id, name = message.get("tool_calls"), message.get("tool_call_id"), message.get("name")
        else:
            role, content = getattr(message, "role", None), getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
            tool_call_id, name = getattr(message, "tool_call_id", None), getattr(message, "name", None)

        blocks = _content_blocks(content)
        text = "\n".join(b.text for b in blocks if isinstance(b, TextMessage))

        if role == "system":
            return cls.system(text)
        if role == "assistant":
            if tool_calls:
                return cls.assistant_tool_calls([ToolCall.from_openai(tc) for tc in tool_calls], text)
            return cls(role=MessageRole.ASSISTANT, content=blocks or [TextMessage(text=text)])
        if role == "tool":
            return cls.tool(tool_call_id or "", text, name=name)
        return cls(role=MessageRole.USER, content=blocks or [TextMessage(text="")])


def _content_blocks(content: Any) -> list[TextMessage | ImageMessage]:
    """Parse OpenAI message ``content`` (str or multimodal list) into Gate content blocks.

    ``image_url`` parts become ``ImageMessage`` (base64 from a ``data:`` URI, else the raw URL).
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [TextMessage(text=content)] if content else []
    blocks: list[TextMessage | ImageMessage] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append(TextMessage(text=part.get("text", "")))
        elif part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "")
            detail = image_url.get("detail")
            if url.startswith("data:") and ";base64," in url:
                blocks.append(ImageMessage(b64=url.split(";base64,", 1)[1], detail=detail))
            elif url:
                blocks.append(ImageMessage(url=url, detail=detail))
    return blocks


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
