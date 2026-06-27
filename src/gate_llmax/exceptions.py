"""Client-side exception hierarchy."""

from __future__ import annotations


class GateError(Exception):
    """Base exception for all Gate client errors."""


class GateAuthError(GateError):
    """Raised when the API key is missing or rejected (HTTP 401)."""


class GateConnectionError(GateError):
    """Raised when the gateway is unreachable."""


class GateTimeoutError(GateError):
    """Raised when the client-side call timeout is exceeded."""


class GateModelNotFoundError(GateError):
    """Raised when the requested model does not exist (HTTP 404)."""


class GateCapabilityError(GateError):
    """Raised when the request uses a feature the model does not support (HTTP 422)."""


class GateServerError(GateError):
    """Raised for unexpected 5xx responses from the gateway."""


class GateBudgetError(GateError):
    """Raised before dispatch when a registered ``.budget()`` check denies the call."""


class GateEscapeHatchWarning(UserWarning):
    """Emitted when calling a discouraged passthrough that bypasses Gate's unified handling."""
