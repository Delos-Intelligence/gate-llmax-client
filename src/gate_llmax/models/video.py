"""Text-to-video request and response models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .request import CallControl
from .response import LLMCallRecord

VideoAspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"]
VideoDuration = Annotated[int, Field(ge=1, le=30, strict=True)]
"""Clip length in seconds; each model accepts a subset (its ``VideoLimits.durations``), Gate rejects the rest."""
VideoResolution = Literal["480p", "720p", "768p", "1080p", "4k"]

VIDEO_EXTENSION_SECONDS = 7
"""Seconds one extension appends to ``source_video``; the provider fixes it, the caller cannot pick."""


class MediaInput(BaseModel):
    """A media file sent to the gateway, by URL (preferred for anything large) or inline base64."""

    url: str | None = Field(default=None, description="An https URL the provider can fetch, e.g. a signed storage URL.")
    data: str | None = Field(default=None, description="Base64-encoded bytes, for small inputs only.")
    mime_type: str | None = Field(default=None, description="MIME type of the file, e.g. ``video/mp4`` or ``audio/mpeg``.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> MediaInput:
        if bool(self.url) == bool(self.data):
            msg = "MediaInput needs exactly one of `url` or `data`."
            raise ValueError(msg)
        return self


class SegmentPoint(BaseModel):
    """A click on the input video for a segmentation model: where, on which frame, and whether it marks the object or the background."""

    x: float = Field(ge=0, le=1, description="Horizontal position, 0 (left) to 1 (right).")
    y: float = Field(ge=0, le=1, description="Vertical position, 0 (top) to 1 (bottom).")
    frame: int = Field(default=0, ge=0, description="Index of the frame the point is on.")
    positive: bool = Field(default=True, description="True marks the object to segment, False the background.")


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

    ``input_video`` / ``input_audio`` / ``speaker_audio`` / ``mask_video`` /
    ``segment_points`` / ``segment_prompt`` feed media models (lip sync, avatars,
    video-to-video, removal, segmentation); the model's ``VideoLimits.task`` says
    which and ``VideoLimits.inputs`` what it takes. There ``start_image`` is the
    portrait or character image, the output length follows the input media, and
    ``prompt`` may be empty.
    """

    model: str
    prompt: str = ""
    aspect_ratio: VideoAspectRatio = "16:9"
    duration_seconds: VideoDuration = 6
    resolution: VideoResolution = "720p"
    with_audio: bool = False
    negative_prompt: str | None = Field(default=None, description="What the clip must not contain, for models that take one.")
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
    input_video: MediaInput | None = Field(
        default=None, description="The clip to lip-sync, transfer motion from, edit, remove from, or segment."
    )
    input_audio: MediaInput | None = Field(default=None, description="The speech for lip sync or a talking avatar.")
    speaker_audio: list[MediaInput] | None = Field(
        default=None, description="One speech track per speaker for a multi-speaker avatar, in the model's speaker order (left to right)."
    )
    mask_video: MediaInput | None = Field(default=None, description="A mask clip for removal: white marks what to remove.")
    segment_points: list[SegmentPoint] | None = Field(default=None, description="Clicks on the object to segment, for segmentation models.")
    segment_prompt: str | None = Field(default=None, description="Text naming what to segment or remove, for models that take one.")


class VideoResponse(LLMCallRecord):
    """Response from ``/v1/videos`` — Gate call metadata + base64 mp4, or its URL when the clip is too large to inline."""

    video: str = Field(default="", description="Base64-encoded video (mp4); for a segmentation model, the mask video.")
    video_url: str | None = Field(
        default=None, description="Where the clip can be downloaded, when Gate hands back a URL instead of (or with) the bytes."
    )
    output_seconds: float | None = Field(default=None, description="Length of the returned clip, when Gate measured it.")
    output_frames: int | None = Field(default=None, description="Frames in the returned clip, when Gate measured it.")
    input_seconds: float | None = Field(default=None, description="Seconds of input video measured, for models priced on their input.")
    video_uri: str | None = Field(
        default=None,
        description=(
            "Provider handle of the generated clip, to pass back as ``VideoSource.uri`` for an extension. "
            "Only while the provider retains the file (Gemini: 2 days, reset by each extension)."
        ),
    )
