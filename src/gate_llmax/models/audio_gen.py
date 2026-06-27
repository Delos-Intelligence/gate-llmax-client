"""Generative audio request/response models — ElevenLabs music, sound effects, dialogue."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .request import CallControl
from .response import BaseAudioResponse

AudioGenMode = Literal["music", "sound_effects", "dialogue"]
AudioMode = Literal["speech", "music", "sound_effects", "dialogue"]
"""Client-facing audio modes: ``speech`` routes to TTS, the rest to generative audio."""


class DialogueTurn(BaseModel):
    """One turn of a multi-speaker dialogue."""

    voice_id: str
    text: str


class AudioGenRequest(CallControl):
    """Generate audio from text via ElevenLabs. ``mode`` selects which generator runs.

    ``prompt`` drives ``music`` and ``sound_effects``; ``inputs`` drives ``dialogue``.
    The other fields are mode-specific knobs, ignored when not applicable.
    """

    model: str
    mode: AudioGenMode
    prompt: str = ""
    music_length_ms: int = 30000
    force_instrumental: bool = False
    duration_seconds: float | None = None
    prompt_influence: float | None = None
    inputs: list[DialogueTurn] = Field(default_factory=list)
    language_code: str | None = None
    output_format: str = "mp3_44100_128"


class AudioGenResponse(BaseAudioResponse):
    """Response from ``/v1/audio/generations`` — Gate call metadata + base64 audio."""

    mode: str = ""
