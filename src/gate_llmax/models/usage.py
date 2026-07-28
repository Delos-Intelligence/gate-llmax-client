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
    reasoning_tokens: int = 0
    finish_reason: str = ""
    reasoning_preview: str = Field(default="", description="Start of the reasoning chain, when content came back empty.")
    request_preview: dict[str, Any] | None = Field(default=None, description="Stripped request: message openings + reasoning knobs.")
    replayable: bool = False


class LatencyRow(BaseModel):
    """Latency aggregates for one model or deployment (and input-size bucket when requested)."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    deployment: str | None = None
    bucket: str | None = Field(default=None, description="Input-token range label, e.g. `2k-20k`; None = all sizes.")
    calls: int = 0
    avg_input: int = 0
    avg_output: int = 0
    ttft_p50: int | None = None
    ttft_p90: int | None = None
    ttft_p99: int | None = None
    dur_p50: int | None = None
    dur_p90: int | None = None
    decode_tps: float | None = Field(default=None, description="Output tokens per second after the first token.")


class TimeseriesPoint(BaseModel):
    """Traffic, failures, median TTFT and spend for one time bucket."""

    model_config = ConfigDict(extra="ignore")

    t: datetime
    calls: int = 0
    failures: int = 0
    ttft_p50: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class StatsRow(BaseModel):
    """Success/failure totals for one model, key or operation."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    calls: int = 0
    failures: int = Field(default=0, description="Non-success calls, client cancellations excluded.")
    error_rate: float = 0.0
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)


class DeploymentInfo(BaseModel):
    """One deployment as the catalog sees it — config only, never credentials."""

    model_config = ConfigDict(extra="ignore")

    deployment: str
    model: str | None = None
    hosting_provider: str = ""
    api_provider: str = ""
    status: str = ""
    priority: int = 1
    provider_model_id: str | None = Field(default=None, description="What actually goes upstream; the name is a label.")
    region: str = ""
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_output_tokens: int | None = None
    input_token_price: float | None = Field(default=None, description="USD/1M override; None inherits the model price.")
    output_token_price: float | None = None
    input_cache_price: float | None = None
    last_error: str | None = None
    status_since: datetime | None = None


class UsageSample(BaseModel):
    """One call as it happened, successes included when asked for."""

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
    duration_ms: int = 0
    ttft_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str = ""
    route: list[dict[str, Any]] | None = Field(default=None, description="Route trace: aliases, fallbacks, retries.")
    request_preview: dict[str, Any] | None = None
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
