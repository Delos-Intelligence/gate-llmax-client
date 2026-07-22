"""Async streaming response helper."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gate_llmax.exceptions import (
    LLMAuthError,
    LLMCapabilityError,
    LLMError,
    LLMModelNotFoundError,
    LLMServerError,
    LLMTimeoutError,
)
from gate_llmax.models.response import StreamChunk

# The gateway cannot change the HTTP status once the 200 SSE headers are flushed, so a terminal
# failure arrives as a final `{"error": {...}}` frame. Mapped to the same exceptions the buffered
# path raises — otherwise a failed stream is indistinguishable from a successful empty one.
ERROR_STATUS_EXCEPTIONS: dict[int, type[LLMError]] = {
    400: LLMCapabilityError,
    401: LLMAuthError,
    403: LLMAuthError,
    404: LLMModelNotFoundError,
    422: LLMCapabilityError,
    504: LLMTimeoutError,
}


def stream_error(payload: dict) -> LLMError:
    """Turn a gateway SSE ``error`` frame into the typed exception the buffered path would raise."""
    error = payload.get("error")
    if not isinstance(error, dict):
        error = {}
    message = str(error.get("message") or payload.get("message") or "Gate stream failed")
    code = error.get("code")
    status = code if isinstance(code, int) else 0
    exception_type = ERROR_STATUS_EXCEPTIONS.get(status, LLMServerError)
    return exception_type(f"{message} (stream error {status or 'unknown'})")


class StreamResponse:
    """Parse SSE lines from a streaming gateway response into `StreamChunk` objects."""

    _response: httpx.Response | None = None

    def __init__(self, response: httpx.Response) -> None:
        """Wrap an open streaming `httpx` response."""
        self._response = response

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Return async iterator over parsed chunks."""
        return self._iter_chunks()

    async def _iter_chunks(self) -> AsyncIterator[StreamChunk]:
        if self._response is None:
            raise ValueError("Response is not open")
        async for line in self._response.aiter_lines():
            new_line = line.strip()
            if not new_line or not new_line.startswith("data: "):
                continue
            payload = new_line[len("data: ") :]
            if payload == "[DONE]":
                return
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "error" in data:
                raise stream_error(data)
            try:
                yield StreamChunk.model_validate(data)
            except ValueError:
                continue
