"""A gateway SSE `error` frame must raise, not read as a successful empty stream."""

from __future__ import annotations

import json

import pytest

from gate_llmax.exceptions import LLMAuthError, LLMCapabilityError, LLMModelNotFoundError, LLMServerError
from gate_llmax.streaming import StreamResponse, stream_error


class FakeResponse:
    """Minimal stand-in for the open httpx streaming response."""

    def __init__(self, lines: list[str]) -> None:
        """Hold the SSE lines this fake will replay."""
        self.lines = lines

    async def aiter_lines(self):
        """Replay the canned SSE lines the way httpx would."""
        for line in self.lines:
            yield line


def frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (400, LLMCapabilityError),
        (401, LLMAuthError),
        (404, LLMModelNotFoundError),
        (422, LLMCapabilityError),
        (502, LLMServerError),
        (None, LLMServerError),
    ],
)
def test_error_frames_map_to_the_typed_exceptions(code, expected):
    exc = stream_error({"error": {"message": "boom", "code": code}})
    assert isinstance(exc, expected)
    assert "boom" in str(exc)


async def test_stream_raises_on_a_terminal_error_frame():
    stream = StreamResponse(
        FakeResponse(
            [
                frame({"text": "hel", "is_done": False}),
                frame({"error": {"message": "bad reasoning_effort", "type": "invalid_request_error", "code": 400}}),
                "data: [DONE]",
            ]
        )
    )
    seen: list[str] = []

    async def drain() -> None:
        async for chunk in stream:
            seen.append(chunk.text)  # noqa: PERF401 - the point is what arrives before the raise

    with pytest.raises(LLMCapabilityError, match="bad reasoning_effort"):
        await drain()
    assert seen == ["hel"]


async def test_a_clean_stream_still_yields_every_chunk():
    stream = StreamResponse(
        FakeResponse([frame({"text": "pong"}), frame({"text": "", "is_done": True, "finish_reason": "stop"}), "data: [DONE]"])
    )
    chunks = [c async for c in stream]
    assert "".join(c.text for c in chunks) == "pong"
    assert chunks[-1].finish_reason == "stop"
