"""Client-side exception hierarchy."""

from __future__ import annotations


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


class LLMServerError(LLMError):
    """Raised for unexpected 5xx responses from the gateway."""


class LLMBudgetError(LLMError):
    """Raised before dispatch when a registered ``.budget()`` check denies the call."""


class LLMEscapeHatchWarning(UserWarning):
    """Emitted when calling a discouraged passthrough that bypasses Gate's unified handling."""
