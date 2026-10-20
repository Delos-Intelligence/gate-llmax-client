"""Image task models (background removal): the ``task`` trait, a prompt-less request, and a PNG whose alpha survives the client.

Self-contained — a MockTransport stands in for the gateway.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import httpx

from gate_llmax import ImageLimits, LLMClient, ModelTraits
from gate_llmax.models.images import ImageRequest

BASE_URL = "https://gate.test"


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def rgba_png(pixels: list[tuple[int, int, int, int]]) -> bytes:
    """A one-row RGBA PNG of ``pixels``."""
    header = struct.pack(">IIBBBBB", len(pixels), 1, 8, 6, 0, 0, 0)
    row = b"\x00" + b"".join(bytes(pixel) for pixel in pixels)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row)) + chunk(b"IEND", b"")


def png_pixels(data: bytes) -> tuple[int, list[tuple[int, ...]]]:
    """The colour type and first-row pixels of an unfiltered 8-bit PNG."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, idat, header = 8, b"", b""
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind, body = data[offset + 4 : offset + 8], data[offset + 8 : offset + 8 + length]
        header = body if kind == b"IHDR" else header
        idat += body if kind == b"IDAT" else b""
        offset += 12 + length
    width, _, _, color_type = struct.unpack(">IIBB", header[:10])
    row = zlib.decompress(idat)[1 : 1 + 4 * width]
    return color_type, [tuple(row[i : i + 4]) for i in range(0, len(row), 4)]


CUTOUT = rgba_png([(255, 0, 0, 255), (0, 0, 0, 0), (12, 34, 56, 128)])


def test_image_limits_task_defaults_to_generation() -> None:
    assert ImageLimits().task is None


def test_image_limits_parse_the_task_from_the_catalog() -> None:
    traits = ModelTraits.model_validate({"image": {"task": "remove_background", "max_reference_images": 1}})
    assert traits.image is not None
    assert (traits.image.task, traits.image.max_reference_images) == ("remove_background", 1)


def test_image_limits_keep_an_unknown_task() -> None:
    assert ImageLimits.model_validate({"task": "upscale"}).task == "upscale"


def test_an_image_request_needs_no_prompt() -> None:
    """A task model works on its input image alone, and an older payload with a prompt still parses."""
    assert ImageRequest(model="bria-rmbg-2.0", images=["QQ=="]).prompt == ""
    assert ImageRequest.model_validate({"model": "gpt-image-1", "prompt": "a cat"}).prompt == "a cat"


def test_the_builder_takes_no_prompt() -> None:
    client = LLMClient(api_key="k", base_url=BASE_URL)
    request = client.image(images=[CUTOUT], operation="studio.remove_background").request
    assert request.prompt == ""
    assert request.images is not None
    assert base64.b64decode(request.images[0]) == CUTOUT


async def test_a_png_keeps_its_alpha_through_the_client() -> None:
    """The input goes out and the cutout comes back byte for byte, transparent pixels and all."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        image = {"b64": base64.b64encode(CUTOUT).decode("ascii"), "output_format": "png"}
        return httpx.Response(200, json={"model": "bria-rmbg-2.0", "status": "SUCCESS", "data": [image]})

    http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    client = LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http)

    response = await client.image(images=[CUTOUT], operation="studio.remove_background").call("bria-rmbg-2.0")

    assert sent[0]["prompt"] == ""
    assert base64.b64decode(sent[0]["images"][0]) == CUTOUT
    out = base64.b64decode(response.data[0].b64)
    assert out == CUTOUT
    assert response.data[0].output_format == "png"
    assert png_pixels(out) == (6, [(255, 0, 0, 255), (0, 0, 0, 0), (12, 34, 56, 128)])
