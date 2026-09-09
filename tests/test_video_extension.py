"""Video extension on the wire: ``source_video`` rides the request, ``video_uri`` comes back on the response.

Self-contained — stubs ``LLMClient._send_video`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, VideoSource
from gate_llmax.models.response import RawUsage
from gate_llmax.models.video import VIDEO_EXTENSION_SECONDS, VideoRequest, VideoResponse
from gate_llmax.types import OutputStatus


def test_source_needs_a_uri_or_bytes() -> None:
    with pytest.raises(ValidationError):
        VideoSource()
    assert VideoSource(uri="files/abc").uri == "files/abc"
    assert VideoSource(data="QQ==").data == "QQ=="


def test_source_video_lands_on_the_request(monkeypatch: Any) -> None:
    sent: list[VideoRequest] = []

    async def _send_video(self: LLMClient, request: VideoRequest) -> VideoResponse:  # noqa: ARG001
        sent.append(request)
        return VideoResponse(model=request.model, video="QQ==", video_uri="files/next", usage=RawUsage(model=request.model))

    monkeypatch.setattr(client_mod.LLMClient, "_send_video", _send_video)
    client = LLMClient(api_key="k", base_url="http://x")
    source = VideoSource(uri="files/abc", deployment_id="dep-1")

    resp = asyncio.run(client.video("keep going", source_video=source, operation="studio.extend").call("veo"))

    assert sent[0].source_video == source
    assert sent[0].model == "veo"
    assert resp.video_uri == "files/next"


def test_a_plain_generation_carries_no_source_and_no_uri() -> None:
    """The extension fields are additive: older gateways and callers never see them set."""
    request = VideoRequest(model="veo", prompt="a wave")
    assert request.source_video is None
    assert "source_video" in request.model_dump()

    response = VideoResponse.model_validate({"model": "veo", "video": "QQ==", "status": OutputStatus.SUCCESS.value})
    assert response.video_uri is None


def test_source_video_round_trips_through_json() -> None:
    request = VideoRequest(model="veo", prompt="onwards", source_video=VideoSource(uri="files/abc", deployment_id="dep-1"))
    again = VideoRequest.model_validate_json(request.model_dump_json())
    assert again.source_video is not None
    assert (again.source_video.uri, again.source_video.deployment_id) == ("files/abc", "dep-1")


def test_extension_length_is_the_providers_not_the_callers() -> None:
    assert VIDEO_EXTENSION_SECONDS == 7
