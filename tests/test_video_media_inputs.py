"""Media inputs on the video surface: lip sync, avatars, video-to-video, removal and segmentation ride ``/v1/videos``.

Self-contained — stubs ``LLMClient._send_video`` so it needs no running gateway.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

import gate_llmax.client as client_mod
from gate_llmax import LLMClient, MediaInput, ModelTraits, SegmentPoint, VideoLimits
from gate_llmax.models.response import RawUsage
from gate_llmax.models.video import VideoRequest, VideoResponse
from gate_llmax.types import OutputStatus


def test_media_input_takes_exactly_one_source() -> None:
    assert MediaInput(url="https://x/clip.mp4").url == "https://x/clip.mp4"
    assert MediaInput(data="QQ==", mime_type="audio/mpeg").data == "QQ=="
    with pytest.raises(ValidationError):
        MediaInput()
    with pytest.raises(ValidationError):
        MediaInput(url="https://x/clip.mp4", data="QQ==")


@pytest.mark.parametrize("point", [{"x": -0.1, "y": 0.5}, {"x": 0.5, "y": 1.1}, {"x": 0.5, "y": 0.5, "frame": -1}])
def test_segment_points_are_normalised(point: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SegmentPoint.model_validate(point)


def test_segment_point_defaults_to_a_positive_click_on_the_first_frame() -> None:
    point = SegmentPoint(x=0.25, y=0.75)
    assert (point.frame, point.positive) == (0, True)


def test_a_media_task_needs_no_prompt() -> None:
    request = VideoRequest(
        model="sync-lipsync-3", input_video=MediaInput(url="https://x/v.mp4"), input_audio=MediaInput(url="https://x/a.mp3")
    )
    assert request.prompt == ""


def test_media_fields_are_additive() -> None:
    """A plain generation carries none of them, and an older payload still parses."""
    request = VideoRequest.model_validate({"model": "veo", "prompt": "a wave"})
    assert request.input_video is None
    assert request.input_audio is None
    assert request.speaker_audio is None
    assert request.mask_video is None
    assert request.segment_points is None
    assert request.segment_prompt is None


def test_media_fields_round_trip_through_json() -> None:
    request = VideoRequest(
        model="sam-3-video",
        input_video=MediaInput(url="https://x/v.mp4", mime_type="video/mp4"),
        mask_video=MediaInput(data="QQ=="),
        segment_points=[SegmentPoint(x=0.1, y=0.2, positive=False)],
        segment_prompt="the dog",
    )
    again = VideoRequest.model_validate_json(request.model_dump_json())
    assert again == request


def test_the_builder_passes_media_inputs_through(monkeypatch: Any) -> None:
    sent: list[VideoRequest] = []

    async def _send_video(self: LLMClient, request: VideoRequest) -> VideoResponse:  # noqa: ARG001
        sent.append(request)
        return VideoResponse(model=request.model, video_url="https://cdn/out.mp4", output_seconds=12.5, usage=RawUsage(model=request.model))

    monkeypatch.setattr(client_mod.LLMClient, "_send_video", _send_video)
    client = LLMClient(api_key="k", base_url="http://x")
    video, audio = MediaInput(url="https://x/v.mp4"), MediaInput(url="https://x/a.mp3")
    speakers = [MediaInput(url="https://x/left.mp3"), MediaInput(url="https://x/right.mp3")]
    points = [SegmentPoint(x=0.5, y=0.5)]

    builder = client.video(
        input_video=video,
        input_audio=audio,
        speaker_audio=speakers,
        mask_video=video,
        segment_points=points,
        segment_prompt="the car",
        operation="studio.lipsync",
    )
    resp = asyncio.run(builder.call("sync-lipsync-3"))

    request = sent[0]
    assert (request.model, request.prompt) == ("sync-lipsync-3", "")
    assert (request.input_video, request.input_audio, request.mask_video) == (video, audio, video)
    assert request.speaker_audio == speakers
    assert (request.segment_points, request.segment_prompt) == (points, "the car")
    assert (resp.video, resp.video_url, resp.output_seconds) == ("", "https://cdn/out.mp4", 12.5)


def test_video_response_url_and_measures_default_to_none() -> None:
    response = VideoResponse.model_validate({"model": "veo", "video": "QQ==", "status": OutputStatus.SUCCESS.value})
    assert response.video_url is None
    assert (response.output_seconds, response.output_frames, response.input_seconds) == (None, None, None)


def test_video_limits_media_traits_default_to_generation() -> None:
    limits = VideoLimits()
    assert (limits.task, limits.inputs, limits.max_input_seconds, limits.speakers) == (None, None, None, None)


def test_video_limits_parse_media_traits_from_the_catalog() -> None:
    traits = ModelTraits.model_validate(
        {"video": {"task": "multi_avatar", "inputs": ["image", "audio"], "max_input_seconds": 50, "speakers": 2}}
    )
    assert traits.video is not None
    assert (traits.video.task, traits.video.inputs, traits.video.max_input_seconds, traits.video.speakers) == (
        "multi_avatar",
        ["image", "audio"],
        50,
        2,
    )


def test_video_limits_keep_an_unknown_task() -> None:
    assert VideoLimits.model_validate({"task": "relight"}).task == "relight"
