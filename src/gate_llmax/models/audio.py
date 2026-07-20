"""Audio transcription request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.request import CallControl
from ..models.response import LLMCallRecord


class AudioRequest(CallControl):
    """Request to transcribe base64-encoded audio."""

    model: str
    audio: str = Field(description="Base64-encoded audio data")
    language: str | None = Field(default=None, description="BCP-47 language code (e.g. 'en', 'fr')")
    response_format: str = Field(default="text", description="Output format: text, json, verbose_json, srt, vtt")
    prompt: str | None = Field(default=None, description="Optional text to bias decoding (domain vocabulary, spelling, style).")
    temperature: float | None = Field(default=None, description="Sampling temperature for transcription (0-1).")


class TranscriptionSegment(BaseModel):
    """One timestamped transcript segment (from ``verbose_json`` STT)."""

    start: float = Field(default=0.0, description="Segment start time in seconds.")
    end: float = Field(default=0.0, description="Segment end time in seconds.")
    text: str = Field(default="", description="Transcript text for this segment.")


class AudioResponse(LLMCallRecord):
    """Response from ``/v1/audio/transcriptions`` — Gate call metadata + transcript text."""

    text: str = ""
    segments: list[TranscriptionSegment] = Field(
        default_factory=list,
        description="Timestamped segments; populated only for ``verbose_json`` on whisper models, empty otherwise.",
    )
