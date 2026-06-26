"""Text-to-video request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .request import CallControl
from .response import GateCallRecord

VideoAspectRatio = Literal["16:9", "9:16"]
VideoDuration = Literal[4, 6, 8]
VideoResolution = Literal["720p", "1080p", "4k"]


class VideoRequest(CallControl):
    """Request to generate a short video from a prompt (optionally image-guided).

    ``start_image`` / ``end_image`` drive image-to-video mode (first/last frame);
    ``reference_images`` provide style/content guidance. Per provider constraints
    the two are mutually exclusive — ``reference_images`` is ignored when a start
    or end frame is supplied.
    """

    model: str
    prompt: str
    aspect_ratio: VideoAspectRatio = "16:9"
    duration_seconds: VideoDuration = 6
    resolution: VideoResolution = "720p"
    with_audio: bool = False
    reference_images: list[str] | None = Field(
        default=None,
        description="Base64-encoded style/content reference images (max 3).",
    )
    start_image: str | None = Field(default=None, description="Base64-encoded first frame.")
    end_image: str | None = Field(default=None, description="Base64-encoded last frame.")


class VideoResponse(GateCallRecord):
    """Response from ``/v1/videos`` — Gate call metadata + base64 mp4."""

    video: str = Field(default="", description="Base64-encoded video (mp4).")
