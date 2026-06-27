"""Async streaming response helper."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from gate_llmax.models.response import StreamChunk


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
                yield StreamChunk.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                continue
