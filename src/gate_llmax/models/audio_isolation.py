"""Audio-isolation request and response models (vocal/background separation)."""

from __future__ import annotations

from pydantic import Field

from .request import CallControl
from .response import LLMCallRecord


class AudioIsolationRequest(CallControl):
    """Request to isolate the foreground voice from a noisy audio clip."""

    model: str
    audio: str = Field(description="Base64-encoded source audio.")
    duration_seconds: float = Field(
        default=0.0,
        description="Length of the source audio in seconds; used only for usage/billing.",
    )


class AudioIsolationResponse(LLMCallRecord):
    """Response from ``/v1/audio/isolation`` — Gate call metadata + base64 isolated audio."""

    audio: str = Field(default="", description="Base64-encoded isolated audio (mp3).")
    response_format: str = "mp3"
