"""Text-to-speech request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .request import CallControl
from .response import BaseAudioResponse

TTSFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class TTSRequest(CallControl):
    """Request to synthesize speech from text."""

    model: str
    text: str
    voice: str = Field(default="alloy", description="Provider-specific voice name; provider validates.")
    response_format: TTSFormat = "mp3"
    speed: float = Field(default=1.0, description="Playback speed multiplier (provider-specific range).")
    speaker_boost: bool = Field(default=False, description="ElevenLabs `use_speaker_boost` — boost similarity to the voice.")


class TTSResponse(BaseAudioResponse):
    """Response from ``/v1/audio/speech`` — Gate call metadata + base64-encoded audio bytes."""

    response_format: str = "mp3"
