"""Core enumerations for the Gate LLM Gateway."""

from __future__ import annotations

from enum import StrEnum

# JSON-serializable values (JSON Schema payloads, tool defs, etc.)
type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type JsonDict = dict[str, JsonValue]


class OutputStatus(StrEnum):
    """Outcome status of a gateway LLM call."""

    SUCCESS = "SUCCESS"
    NO_PARSE = "NO_PARSE"
    NO_VALIDATION = "NO_VALIDATION"
    EMPTY = "EMPTY"
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    NO_CONNECT = "NO_CONNECT"
    NO_DEPLOYMENT = "NO_DEPLOYMENT"
    NO_API_KEY = "NO_API_KEY"
    CANCELLED = "CANCELLED"


class ReasoningEffort(StrEnum):
    """Reasoning effort level for models that support extended thinking."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
