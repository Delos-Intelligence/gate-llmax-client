"""Client-side exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.response import ContentFilter


class LLMError(Exception):
    """Base exception for all Gate client errors."""


class LLMAuthError(LLMError):
    """Raised when the API key is missing or rejected (HTTP 401)."""


class LLMConnectionError(LLMError):
    """Raised when the gateway is unreachable."""


class LLMTimeoutError(LLMError):
    """Raised when the client-side call timeout is exceeded."""


class LLMModelNotFoundError(LLMError):
    """Raised when the requested model does not exist (HTTP 404)."""


class LLMCapabilityError(LLMError):
    """Raised when the request uses a feature the model does not support (HTTP 422)."""


class LLMContentFilterError(LLMError):
    """Raised when the upstream safety/content filter rejected the prompt (HTTP 400); ``content_filter`` says why when known."""

    def __init__(self, message: str, content_filter: ContentFilter | None = None) -> None:
        """Keep the structured verdict beside the message."""
        super().__init__(message)
        self.content_filter = content_filter


class LLMContextOverflowError(LLMError):
    """Raised when the prompt exceeds the model's context window (HTTP 413)."""


class LLMServerError(LLMError):
    """Raised for unexpected 5xx responses from the gateway."""


class LLMEmptyResponseError(LLMServerError):
    """Raised when every model the gateway tried finished without visible text or a tool call."""


class LLMBudgetError(LLMError):
    """Raised before dispatch when a registered ``.budget()`` check denies the call."""


class LLMEscapeHatchWarning(UserWarning):
    """Emitted when calling a discouraged passthrough that bypasses Gate's unified handling."""
