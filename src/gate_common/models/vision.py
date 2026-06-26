"""Vision OCR request and result models (Azure AI Vision)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VisionOCRRequest(BaseModel):
    """Request to run OCR over one or more base64-encoded images.

    The retry/timeout fields mirror ``CallControl``; they are inlined rather than
    inherited because this module is a leaf imported by ``response`` (inheriting
    would create an import cycle through ``request``/``messages``).
    """

    model: str
    images: list[str] = Field(default_factory=list, description="Base64-encoded images to OCR.")
    max_tries: int | None = Field(default=None, description="Per-call upstream attempts, including the first.")
    timeout: int | None = Field(default=None, description="Per-call upstream timeout in seconds.")


class VisionPoint(BaseModel):
    """A single (x, y) pixel position."""

    x: int
    y: int


class VisionWord(BaseModel):
    """A single word recognised by Azure Vision OCR."""

    text: str
    confidence: float = 0.0
    bounding_polygon: list[VisionPoint] = Field(default_factory=list)


class VisionLine(BaseModel):
    """A line of text recognised by Azure Vision OCR."""

    text: str
    bounding_polygon: list[VisionPoint] = Field(default_factory=list)
    words: list[VisionWord] = Field(default_factory=list)


class VisionOCR(BaseModel):
    """Full OCR result returned by the vision endpoint.

    ``lines`` is a flat list across all blocks; ``angle`` is the dominant
    text rotation (rounded to the nearest 90°, normalised to 0–359).
    """

    lines: list[VisionLine] = Field(default_factory=list)
    angle: int = 0
