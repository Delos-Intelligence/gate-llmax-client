"""Audio transcription request and response models."""

from __future__ import annotations

from pydantic import Field

from ..models.request import CallControl
from ..models.response import GateCallRecord


class AudioRequest(CallControl):
    """Request to transcribe base64-encoded audio."""

    model: str
    audio: str = Field(description="Base64-encoded audio data")
    language: str | None = Field(default=None, description="BCP-47 language code (e.g. 'en', 'fr')")
    response_format: str = Field(default="text", description="Output format: text, json, verbose_json, srt, vtt")
    prompt: str | None = Field(default=None, description="Optional text to bias decoding (domain vocabulary, spelling, style).")
    temperature: float | None = Field(default=None, description="Sampling temperature for transcription (0-1).")


class AudioResponse(GateCallRecord):
    """Response from ``/v1/audio/transcriptions`` — Gate call metadata + transcript text."""

    text: str = ""
