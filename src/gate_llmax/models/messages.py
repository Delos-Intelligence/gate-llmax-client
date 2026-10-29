"""Message types for Gate requests (chat roles + text / image / audio / video content blocks)."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import JsonDict
from .response import ReasoningBlock, ToolCall

logger = logging.getLogger(__name__)


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
    media_type: str | None = Field(
        default=None,
        description="MIME type of the image (e.g. image/png); the Gate server still sniffs the bytes if unset.",
    )
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


class AudioMessage(BaseModel):
    """Audio segment as raw base64 (no data: prefix); only models with the ``audio_input`` trait take it."""

    model_config = ConfigDict(extra="forbid")

    b64: str
    format: str = Field(default="wav", description="Container of the bytes: wav or mp3 travel everywhere audio input does.")


class VideoMessage(BaseModel):
    """Video segment: exactly one of raw base64 or a URL; only models with the ``video_input`` trait take it."""

    model_config = ConfigDict(extra="forbid")

    b64: str | None = None
    url: str | None = None
    media_type: str = Field(default="video/mp4", description="MIME type of the video bytes.")

    @model_validator(mode="after")
    def exactly_one_video_source(self) -> Self:
        """Validate that exactly one of b64 or url is provided."""
        if bool(self.b64) == bool(self.url):
            msg = "VideoMessage requires exactly one of b64 or url (non-empty)"
            raise ValueError(msg)
        return self


type ContentBlock = TextMessage | ImageMessage | AudioMessage | VideoMessage


class ToolChangeAction(StrEnum):
    """Whether a tool joins or leaves the conversation."""

    ADD = "add"
    REMOVE = "remove"


class ToolChange(BaseModel):
    """One tool, named as in ``LLMRequest.tools``, added to or removed from the model's tool set from this turn on."""

    model_config = ConfigDict(extra="forbid")

    action: ToolChangeAction
    name: str


class Message(BaseModel):
    """A single turn in a conversation."""

    role: MessageRole
    content: list[ContentBlock] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="OpenAI-shaped tool calls on an assistant turn.",
    )
    reasoning: str | None = Field(default=None, description="Assistant thinking text, replayed to the model on the next turn.")
    reasoning_signature: str | None = Field(default=None, description="Opaque signature validating a replayed Anthropic thinking block.")
    reasoning_blocks: list[ReasoningBlock] | None = Field(
        default=None, description="An assistant turn's reasoning blocks in output order; supersedes reasoning/reasoning_signature."
    )
    tool_changes: list[ToolChange] | None = Field(
        default=None, description="On a system turn: tools added or removed here; ``LLMRequest.tools`` then lists every tool ever offered."
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
    def tools_changed(cls, added: list[str] | None = None, removed: list[str] | None = None) -> Message:
        """Create a system turn adding and removing tools by name, keeping the cached prefix where the model allows it."""
        changes = [ToolChange(action=ToolChangeAction.ADD, name=n) for n in added or []]
        changes += [ToolChange(action=ToolChangeAction.REMOVE, name=n) for n in removed or []]
        return cls(role=MessageRole.SYSTEM, tool_changes=changes)

    @classmethod
    def assistant_tool_calls(cls, tool_calls: list[ToolCall], content: str = "") -> Message:
        """Create an assistant turn that requested ``tool_calls`` (optional accompanying text)."""
        blocks: list[ContentBlock] = [TextMessage(text=content)] if content else []
        return cls(role=MessageRole.ASSISTANT, content=blocks, tool_calls=tool_calls)

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        content: str,
        name: str | None = None,
        images: list[ImageMessage] | None = None,
    ) -> Message:
        """Create a tool-result message answering a prior tool call, with the images the tool returned."""
        blocks: list[ContentBlock] = [TextMessage(text=content), *(images or [])]
        return cls(role=MessageRole.TOOL, content=blocks, tool_call_id=tool_call_id, name=name)

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
            reasoning, reasoning_signature = message.get("reasoning"), message.get("reasoning_signature")
            reasoning_blocks = message.get("reasoning_blocks")
            tool_changes = message.get("tool_changes")
        else:
            role, content = getattr(message, "role", None), getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
            tool_call_id, name = getattr(message, "tool_call_id", None), getattr(message, "name", None)
            reasoning, reasoning_signature = getattr(message, "reasoning", None), getattr(message, "reasoning_signature", None)
            reasoning_blocks = getattr(message, "reasoning_blocks", None)
            tool_changes = getattr(message, "tool_changes", None)

        blocks = _content_blocks(content)
        text = "\n".join(b.text for b in blocks if isinstance(b, TextMessage))

        if role == "system":
            msg = cls.system(text)
            if tool_changes:
                msg.tool_changes = [ToolChange.model_validate(c, from_attributes=True) for c in tool_changes]
            return msg
        if role == "assistant":
            if tool_calls:
                msg = cls.assistant_tool_calls([ToolCall.from_openai(tc) for tc in tool_calls], text)
            else:
                msg = cls(role=MessageRole.ASSISTANT, content=blocks or [TextMessage(text=text)])
            msg.reasoning, msg.reasoning_signature = reasoning, reasoning_signature
            if reasoning_blocks is not None:
                msg.reasoning_blocks = [ReasoningBlock.model_validate(b, from_attributes=True) for b in reasoning_blocks]
            return msg
        if role == "tool":
            # Tool results may carry images (e.g. a screenshot/read_image tool) — keep blocks.
            return cls(role=MessageRole.TOOL, content=blocks or [TextMessage(text="")], tool_call_id=tool_call_id or "", name=name)
        return cls(role=MessageRole.USER, content=blocks or [TextMessage(text="")])


def _content_blocks(content: Any) -> list[ContentBlock]:
    """Parse OpenAI message ``content`` (str or multimodal list) into Gate content blocks.

    ``image_url`` parts become ``ImageMessage`` (base64 from a ``data:`` URI, else the raw URL).
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [TextMessage(text=content)] if content else []
    blocks: list[ContentBlock] = []
    for part in content:
        if not isinstance(part, dict):
            logger.warning("ignoring non-dict content part: %r", type(part).__name__)
        elif (block := _content_block(part)) is not None:
            blocks.append(block)
        else:
            logger.warning("ignoring unparseable %r content part", part.get("type"))
    return blocks


def _data_uri(url: str) -> tuple[str, str] | None:
    """(media type, base64) of a ``data:`` URI, or None for a plain URL."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, b64 = url.split(";base64,", 1)
    return header.removeprefix("data:"), b64


def _content_block(part: dict[str, Any]) -> ContentBlock | None:
    """One OpenAI or Anthropic content part as a Gate block; video may ride ``image_url`` as Gemini's wire spells it."""
    part_type = part.get("type")
    if part_type == "text":
        return TextMessage(text=part.get("text", ""))
    if part_type == "input_audio":
        audio = part.get("input_audio") or {}
        return AudioMessage(b64=audio["data"], format=audio.get("format") or "wav") if audio.get("data") else None
    if part_type == "image":
        return _anthropic_image_block(part.get("source") or {})
    if part_type in ("image_url", "video_url"):
        return _url_block(part_type, part.get(part_type) or {})
    return None


def _url_block(part_type: str, ref: dict[str, Any]) -> ImageMessage | VideoMessage | None:
    """An ``image_url`` / ``video_url`` part, inline or by URL; an inline ``video/*`` payload is video whatever the part says."""
    url = ref.get("url", "")
    if not url:
        return None
    inline = _data_uri(url)
    if part_type == "video_url" or (inline is not None and inline[0].startswith("video/")):
        return VideoMessage(b64=inline[1], media_type=inline[0] or "video/mp4") if inline else VideoMessage(url=url)
    if inline is not None:
        return ImageMessage(b64=inline[1], media_type=inline[0] or None, detail=ref.get("detail"))
    return ImageMessage(url=url, detail=ref.get("detail"))


def _anthropic_image_block(source: dict[str, Any]) -> ImageMessage | None:
    """Parse an Anthropic-style image ``source`` ({"type": "base64"|"url", ...}) into an ImageMessage."""
    if source.get("type") == "base64" and source.get("data"):
        return ImageMessage(b64=source["data"], media_type=source.get("media_type"))
    if source.get("type") == "url" and source.get("url"):
        return ImageMessage(url=source["url"])
    return None


def content_block_to_openai(block: ContentBlock) -> JsonDict:
    """Map a Gate content block to an OpenAI chat `content` element; video rides ``image_url``, the form Gemini's OpenAI wire reads."""
    if isinstance(block, TextMessage):
        return {"type": "text", "text": block.text}
    if isinstance(block, AudioMessage):
        return {"type": "input_audio", "input_audio": {"data": block.b64, "format": block.format}}
    if isinstance(block, VideoMessage):
        return {"type": "image_url", "image_url": {"url": block.url or f"data:{block.media_type};base64,{block.b64}"}}
    detail = block.detail or "high"
    if block.url is not None:
        return {"type": "image_url", "image_url": {"url": block.url, "detail": detail}}
    media_type = block.media_type or "image/jpeg"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{block.b64}", "detail": detail},
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
    if msg.reasoning is not None:
        payload["reasoning"] = msg.reasoning
    if msg.reasoning_signature is not None:
        payload["reasoning_signature"] = msg.reasoning_signature
    if msg.reasoning_blocks is not None:
        payload["reasoning_blocks"] = [block.model_dump() for block in msg.reasoning_blocks]
    if msg.tool_changes is not None:
        payload["tool_changes"] = [change.model_dump(mode="json") for change in msg.tool_changes]
    if msg.tool_call_id is not None:
        payload["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        payload["name"] = msg.name
    return payload


def message_plain_text(content: str | list[ContentBlock]) -> str:
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
