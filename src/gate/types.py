"""Re-exports of shared types for SDK consumers.

Import from `gate` directly for the public API; this module is for internal use.
"""

from gate_common.models.config import DeploymentInfo, DeploymentStatus, ModelCapabilities, ModelInfo
from gate_common.models.messages import (
    ImageMessage,
    Message,
    MessageRole,
    TextMessage,
    content_block_to_openai,
    last_message_plain_text,
    message_plain_text,
    message_to_openai_dict,
)
from gate_common.models.request import GateRequest, RequestSpecifics
from gate_common.models.response import GateResponse, RawUsage, StreamChunk
from gate_common.types import JsonDict, JsonValue, OutputStatus, ReasoningEffort

__all__ = [
    "DeploymentInfo",
    "DeploymentStatus",
    "GateRequest",
    "GateResponse",
    "ImageMessage",
    "JsonDict",
    "JsonValue",
    "Message",
    "MessageRole",
    "ModelCapabilities",
    "ModelInfo",
    "OutputStatus",
    "RawUsage",
    "ReasoningEffort",
    "RequestSpecifics",
    "StreamChunk",
    "TextMessage",
    "content_block_to_openai",
    "last_message_plain_text",
    "message_plain_text",
    "message_to_openai_dict",
]
