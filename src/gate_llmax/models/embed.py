"""Embedding request and response models.

``EmbedResponse`` keeps OpenAI's ``object: "list"`` / ``data`` / ``model`` shape
for wire compat but also carries Gate call metadata (``deployment_id``,
``status``, ``latency_ms``) via ``GateCallRecord``. OpenAI clients ignore the
extra fields, so this remains drop-in compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .request import CallControl
from .response import GateCallRecord


class EmbedRequest(CallControl):
    """Request to embed one or more strings."""

    model: str
    input: str | list[str]

    @field_validator("input")
    @classmethod
    def _strip_newlines(cls, value: str | list[str]) -> str | list[str]:
        """Replace newlines with spaces (OpenAI's recommendation for embedding inputs)."""
        if isinstance(value, str):
            return value.replace("\n", " ")
        return [v.replace("\n", " ") for v in value]


class EmbedObject(BaseModel):
    """Single embedding vector (mirrors OpenAI ``Embedding``)."""

    index: int = 0
    object: Literal["embedding"] = "embedding"
    embedding: list[float] = Field(default_factory=list)


class EmbedResponse(GateCallRecord):
    """Response from ``/v1/embeddings`` — Gate call metadata + OpenAI-style ``data``."""

    object: Literal["list"] = "list"
    data: list[EmbedObject] = Field(default_factory=list)
