"""Video request bounds and the per-model capability traits Gate validates them against."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gate_llmax import ImageLimits, LLMClient, ModelTraits, VideoLimits
from gate_llmax.models.video import VideoRequest


def test_defaults_are_unchanged() -> None:
    request = VideoRequest(model="veo", prompt="a wave")
    assert (request.aspect_ratio, request.duration_seconds, request.resolution) == ("16:9", 6, "720p")


@pytest.mark.parametrize("ratio", ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"])
def test_wider_aspect_ratios_are_accepted(ratio: str) -> None:
    assert VideoRequest.model_validate({"model": "m", "prompt": "p", "aspect_ratio": ratio}).aspect_ratio == ratio


@pytest.mark.parametrize("resolution", ["480p", "720p", "768p", "1080p", "4k"])
def test_wider_resolutions_are_accepted(resolution: str) -> None:
    assert VideoRequest.model_validate({"model": "m", "prompt": "p", "resolution": resolution}).resolution == resolution


@pytest.mark.parametrize("seconds", [1, 4, 6, 8, 15, 30])
def test_any_duration_from_one_to_thirty_seconds(seconds: int) -> None:
    assert VideoRequest(model="m", prompt="p", duration_seconds=seconds).duration_seconds == seconds


@pytest.mark.parametrize("field_value", [{"duration_seconds": 0}, {"duration_seconds": 31}, {"aspect_ratio": "2:1"}, {"resolution": "2k"}])
def test_out_of_range_values_are_rejected(field_value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        VideoRequest.model_validate({"model": "m", "prompt": "p", **field_value})


def test_the_builder_carries_the_wider_values() -> None:
    client = LLMClient(api_key="k", base_url="http://x")
    builder = client.video("a cat", aspect_ratio="21:9", duration_seconds=12, resolution="768p", operation="t")
    assert (builder.request.aspect_ratio, builder.request.duration_seconds, builder.request.resolution) == ("21:9", 12, "768p")


def test_video_limits_default_to_unknown_capabilities() -> None:
    limits = VideoLimits()
    assert limits.aspect_ratios is None
    assert limits.audio is None


def test_video_limits_parse_from_catalog_attributes() -> None:
    traits = ModelTraits.model_validate(
        {"video": {"durations": [5, 10], "resolutions": ["480p", "768p"], "aspect_ratios": ["16:9", "1:1"], "audio": "always"}}
    )
    assert traits.video is not None
    assert traits.video.aspect_ratios == ["16:9", "1:1"]
    assert traits.video.audio == "always"


def test_video_limits_reject_an_unknown_audio_mode() -> None:
    with pytest.raises(ValidationError):
        VideoLimits.model_validate({"audio": "sometimes"})


def test_image_limits_qualities() -> None:
    assert ImageLimits().qualities is None
    assert ImageLimits(qualities=["low", "medium", "high"]).qualities == ["low", "medium", "high"]


def test_video_limits_min_duration_by_resolution() -> None:
    assert VideoLimits().min_duration_by_resolution is None
    traits = ModelTraits.model_validate({"video": {"resolutions": ["720p", "1080p"], "min_duration_by_resolution": {"1080p": 8}}})
    assert traits.video is not None
    assert traits.video.min_duration_by_resolution == {"1080p": 8}
