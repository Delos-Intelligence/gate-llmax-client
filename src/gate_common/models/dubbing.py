"""Dubbing request and response models (translate spoken audio to another language)."""

from __future__ import annotations

from pydantic import Field

from .request import CallControl
from .response import GateCallRecord


class DubbingRequest(CallControl):
    """Request to dub a remote audio/video source into ``target_lang``.

    The source is referenced by URL rather than uploaded inline: dubbing providers
    fetch and transcode it server-side, and clips are typically too large to base64.
    """

    model: str
    source_url: str = Field(description="Publicly reachable URL of the source media.")
    source_lang: str = Field(description="ISO-639-1 code of the source audio (e.g. 'en', 'fr').")
    target_lang: str = Field(description="ISO-639-1 code to dub into.")
    duration_seconds: float = Field(
        default=0.0,
        description="Length of the source media in seconds; used only for usage/billing.",
    )
    watermark: bool = False


class DubbingResponse(GateCallRecord):
    """Response from ``/v1/audio/dubbing`` — Gate call metadata + base64 dubbed audio."""

    audio: str = Field(default="", description="Base64-encoded dubbed audio (mp3).")
    target_lang: str = ""
