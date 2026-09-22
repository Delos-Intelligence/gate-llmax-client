"""OCR request and response models: extract text (per-page markdown) from a document or image."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..types import JsonDict
from .request import CallControl
from .response import LLMCallRecord


class OCRDocument(BaseModel):
    """The thing to OCR: a remote URL or an inline base64 payload, either a document (PDF) or a single image."""

    type: Literal["document_url", "image_url", "document_base64", "image_base64"]
    url: str | None = Field(default=None, description="Set for document_url / image_url.")
    data: str | None = Field(default=None, description="Base64 payload set for document_base64 / image_base64.")


class OCRRequest(CallControl):
    """Request to OCR one document or image into per-page markdown."""

    model: str
    document: OCRDocument
    pages: list[int] | None = Field(default=None, description="0-indexed pages to process; None = all.")
    include_images: bool = Field(default=False, description="Return extracted image crops alongside the text.")


class OCRPage(BaseModel):
    """One page of the result."""

    index: int
    markdown: str = ""
    images: list[JsonDict] = Field(default_factory=list)
    dimensions: JsonDict | None = None


class OCRResponse(LLMCallRecord):
    """Response from ``/v1/ocr`` — Gate call metadata plus the extracted pages."""

    pages: list[OCRPage] = Field(default_factory=list)
    pages_processed: int = 0
