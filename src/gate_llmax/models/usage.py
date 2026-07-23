"""Wire models for the dev-key usage insight routes (`GET /v1/usage/*`).

What failed on the gateway, why, and the request body needed to reproduce it. Read-only,
and gated on a ``dev`` API key — a plain consumer key gets 403.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyName(BaseModel):
    """An API key as the outside world refers to it. Never carries the secret."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    is_active: bool = True


class ErrorGroup(BaseModel):
    """Failures sharing a status / model / key / operation / transport.

    ``sample_detail`` is the most recent provider message for the group — the difference
    between "278 capability errors" and "278 x array too long, expected maximum 128".
    ``sample_log_id`` is a log whose request body was captured, ready for ``get_payload``.
    """

    model_config = ConfigDict(extra="ignore")

    status: str
    model: str | None = None
    api_key: str | None = None
    operation: str = ""
    request_type: str = ""
    calls: int = 0
    cost: float = 0.0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    sample_detail: str = ""
    sample_log_id: str | None = None
    replayable: int = Field(default=0, description="How many failures in this group kept their request body.")


class ErrorReport(BaseModel):
    """What failed in a window, grouped, worst first."""

    model_config = ConfigDict(extra="ignore")

    window_from: datetime
    window_to: datetime
    keys: list[str] = Field(default_factory=list, description="Key names the window was filtered to; empty = all keys.")
    total_failures: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    groups: list[ErrorGroup] = Field(default_factory=list)


class ErrorSample(BaseModel):
    """One failed call, as it happened."""

    model_config = ConfigDict(extra="ignore")

    id: str
    created_at: datetime
    status: str
    detail: str = ""
    model: str | None = None
    api_key: str | None = None
    deployment: str | None = None
    operation: str = ""
    request_type: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    replayable: bool = False


class StoredPayload(BaseModel):
    """The request body of a failed call, ready to POST back at ``endpoint``.

    Base64 attachments were replaced by size markers when the row was written, so a payload
    with images replays as text unless they are re-attached.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    created_at: datetime
    status: str
    detail: str = ""
    model: str | None = None
    request_type: str = ""
    endpoint: str
    payload: dict[str, Any] = Field(default_factory=dict)
