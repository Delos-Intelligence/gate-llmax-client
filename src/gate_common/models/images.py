"""Image generation / edit request and response models.

A single ``ImageRequest`` covers both pure text-to-image and image-edit modes.
The route picks edit vs generate by whether ``images`` is set: when non-empty,
the request is interpreted as "edit these images according to ``prompt``".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .request import CallControl
from .response import GateCallRecord

AspectRatio = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
ImageQuality = Literal["low", "medium", "high", "auto"]
ImageSize = tuple[int, int]
"""``(width, height)`` in pixels. ``None`` on a request means provider default (auto)."""

_ASPECT_RATIO_TO_SIZE: dict[AspectRatio, ImageSize] = {
    "1:1": (1024, 1024),
    "2:3": (1024, 1536),
    "3:4": (1024, 1536),
    "4:5": (1024, 1536),
    "9:16": (1024, 1536),
    "3:2": (1536, 1024),
    "4:3": (1536, 1024),
    "5:4": (1536, 1024),
    "16:9": (1536, 1024),
    "21:9": (1536, 1024),
}


class ImageRequest(CallControl):
    """Request to generate or edit images.

    Edit mode is triggered by ``images`` being non-empty. ``mask`` is OpenAI-only
    and ignored by Gemini. Provider-specific knobs (``background``,
    ``output_format``, ``output_compression``) are honored by OpenAI ``gpt-image-1``
    and ignored (with a warning) by Gemini.
    """

    model: str
    prompt: str
    images: list[str] | None = Field(
        default=None,
        description="Base64-encoded input images. When set, the call is an edit; otherwise a fresh generation.",
    )
    mask: str | None = Field(
        default=None,
        description="Base64-encoded mask (OpenAI only). Ignored by other providers.",
    )
    n: int = 1
    quality: ImageQuality = "medium"
    size: ImageSize | None = Field(
        default=(1024, 1024),
        description="(width, height) in pixels; ``None`` lets the provider choose (auto).",
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None,
        description=(
            "Convenience override for `size`. When set, populates `size` from a fixed mapping "
            "(portrait ratios -> 1024x1536, landscape ratios -> 1536x1024, 1:1 -> 1024x1024)."
        ),
    )
    background: Literal["transparent", "opaque", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = None
    partial_images: int = Field(
        default=3,
        description="Number of intermediate frames for streaming (OpenAI only). Ignored when calling /v1/images.",
    )

    @model_validator(mode="after")
    def _apply_aspect_ratio(self) -> ImageRequest:
        if self.aspect_ratio is not None:
            self.size = _ASPECT_RATIO_TO_SIZE[self.aspect_ratio]
        return self


class ImageData(BaseModel):
    """One generated/edited image, base64-encoded."""

    b64: str = Field(description="Base64-encoded image bytes.")
    output_format: str | None = Field(
        default=None,
        description="Echoed back when the client requested a specific output format.",
    )


class ImageResponse(GateCallRecord):
    """Response from ``/v1/images`` — Gate call metadata + generated images."""

    data: list[ImageData] = Field(default_factory=list)
