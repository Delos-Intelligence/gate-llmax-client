"""Text-to-video request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .request import CallControl
from .response import LLMCallRecord

VideoAspectRatio = Literal["16:9", "9:16"]
VideoDuration = Literal[4, 6, 8]
VideoResolution = Literal["720p", "1080p", "4k"]

VIDEO_EXTENSION_SECONDS = 7
"""Seconds one extension appends to ``source_video``; the provider fixes it, the caller cannot pick."""


class VideoSource(BaseModel):
    """A previously generated clip to extend, named the way the gateway handed it back.

    ``uri`` is the provider's own handle (``VideoResponse.video_uri``) — the only form the
    Gemini Developer API accepts, and it lives in the project of the deployment that made
    it, hence ``deployment_id``. ``data`` carries the mp4 itself, for Vertex deployments.
    """

    uri: str | None = Field(default=None, description="Provider file URI of a previous generation (``VideoResponse.video_uri``).")
    data: str | None = Field(default=None, description="Base64-encoded mp4 of the clip to extend; Vertex deployments only.")
    deployment_id: str | None = Field(
        default=None,
        description="Deployment that produced ``uri``; the file is only reachable with that deployment's credential.",
    )

    @model_validator(mode="after")
    def _needs_a_clip(self) -> VideoSource:
        if not self.uri and not self.data:
            msg = "VideoSource needs `uri` or `data`."
            raise ValueError(msg)
        return self


class VideoRequest(CallControl):
    """Request to generate a short video from a prompt (optionally image-guided).

    ``start_image`` / ``end_image`` drive image-to-video mode (first/last frame);
    ``reference_images`` provide style/content guidance. Per provider constraints
    the two are mutually exclusive — ``reference_images`` is ignored when a start
    or end frame is supplied.

    ``source_video`` switches to extension: the clip grows by
    ``VIDEO_EXTENSION_SECONDS`` at 720p, continuing from its last second, and
    ``duration_seconds`` / ``resolution`` / the image inputs are ignored.
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
    source_video: VideoSource | None = Field(
        default=None,
        description="Clip to extend instead of generating anew; the output is the whole clip plus ~7s, at 720p.",
    )


class VideoResponse(LLMCallRecord):
    """Response from ``/v1/videos`` — Gate call metadata + base64 mp4."""

    video: str = Field(default="", description="Base64-encoded video (mp4).")
    video_uri: str | None = Field(
        default=None,
        description=(
            "Provider handle of the generated clip, to pass back as ``VideoSource.uri`` for an extension. "
            "Only while the provider retains the file (Gemini: 2 days, reset by each extension)."
        ),
    )
