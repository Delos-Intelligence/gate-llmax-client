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
    diarize: bool = Field(default=False, description="Label each segment with a speaker; supported on Mistral voxtral only.")


class TranscriptionSegment(BaseModel):
    """One timestamped transcript segment."""

    start: float = Field(default=0.0, description="Segment start time in seconds.")
    end: float = Field(default=0.0, description="Segment end time in seconds.")
    text: str = Field(default="", description="Transcript text for this segment.")
    speaker: str | None = Field(default=None, description="Speaker label, set only when ``diarize`` was requested.")
    avg_logprob: float | None = Field(default=None, description="Whisper's mean token log-probability for the segment.")
    no_speech_prob: float | None = Field(default=None, description="Whisper's probability that the segment is silence.")
    compression_ratio: float | None = Field(default=None, description="Whisper's gzip ratio; a high value flags a repetition loop.")


class AudioResponse(LLMCallRecord):
    """Response from ``/v1/audio/transcriptions`` — Gate call metadata + transcript text."""

    text: str = ""
    language: str | None = Field(default=None, description="Language the provider detected or was told, when it reports one.")
    segments: list[TranscriptionSegment] = Field(
        default_factory=list,
        description="Timestamped segments; empty unless the provider reports them.",
    )
