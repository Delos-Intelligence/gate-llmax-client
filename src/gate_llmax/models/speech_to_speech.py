"""Speech-to-speech request and response models (voice changer: keep the performance, swap the timbre)."""

from __future__ import annotations

from pydantic import Field

from .request import CallControl
from .response import LLMCallRecord


class SpeechToSpeechRequest(CallControl):
    """Request to re-voice a spoken clip, preserving its delivery.

    Unlike TTS, the source is a performance: the model keeps the timing, accent and
    emotion of ``audio`` and only substitutes the timbre of ``voice``.
    """

    model: str
    audio: str = Field(description="Base64-encoded source audio.")
    voice: str = Field(description="Target voice id; the same ids TTS accepts.")
    duration_seconds: float = Field(
        default=0.0,
        description="Length of the source audio in seconds; used only for usage/billing.",
    )
    remove_background_noise: bool = Field(
        default=False,
        description="Run the source through voice isolation before converting.",
    )
    seed: int | None = Field(
        default=None,
        description="Best-effort deterministic sampling; 0 to 4294967295.",
    )
    output_format: str = "mp3_44100_128"


class SpeechToSpeechResponse(LLMCallRecord):
    """Response from ``/v1/audio/speech-to-speech`` — Gate call metadata + base64 converted audio."""

    audio: str = Field(default="", description="Base64-encoded converted audio (mp3).")
    voice: str = ""
    response_format: str = "mp3"
