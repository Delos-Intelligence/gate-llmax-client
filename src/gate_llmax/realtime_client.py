"""Client-side realtime session: an async-context websocket to Gate's ``/v1/realtime``, used like any other modality."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Self

from .models.realtime import (
    RealtimeAudioData,
    RealtimeCapabilities,
    RealtimeEvent,
    RealtimeEventType,
    RealtimeSessionConfig,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

# The three realtime surfaces, keyed by their session purpose shorthand.
PURPOSE_ENDPOINT = {"converse": "converse", "transcribe": "transcribe", "speak": "speak"}


class RealtimeConnectionError(Exception):
    """The realtime websocket could not be opened or the session failed to start."""


class RealtimeConnection:
    """One live-voice session: push audio/video in, iterate ``RealtimeEvent``s out."""

    def __init__(self, ws_url: str, api_key: str, config: RealtimeSessionConfig, *, output_sample_rate: int) -> None:
        """Bind the resolved ws URL, auth key and session config; the socket opens in ``__aenter__``."""
        self._ws_url = ws_url
        self._api_key = api_key
        self._config = config
        self._output_sample_rate = output_sample_rate
        self._ws: ClientConnection | None = None
        self.capabilities = RealtimeCapabilities()

    async def __aenter__(self) -> Self:
        """Open the socket, read the capabilities handshake, and send the session config."""
        from websockets.asyncio.client import connect

        try:
            self._ws = await connect(self._ws_url, additional_headers={"X-Gate-Key": self._api_key})
            created = json.loads(await self._ws.recv())
            self.capabilities = RealtimeCapabilities.model_validate(created.get("capabilities", {}))
            await self._ws.send(json.dumps({"type": "session.update", "session": self._config.model_dump(exclude_none=True)}))
        except Exception as exc:
            raise RealtimeConnectionError(str(exc)) from exc
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the socket on context exit."""
        await self.aclose()

    @property
    def connected(self) -> bool:
        """Whether the socket is open."""
        return self._ws is not None

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps(payload))

    async def push_audio(self, pcm: bytes) -> None:
        """Send a PCM16 audio chunk (binary frame)."""
        if self._ws is not None:
            await self._ws.send(pcm)

    async def send_video(self, jpeg: bytes, source: int = 0) -> None:
        """Send one JPEG video frame (no-op server-side unless ``capabilities.video_input``)."""
        await self._send({"type": "video", "data": base64.b64encode(jpeg).decode(), "source": source})

    async def send_text(self, text: str) -> None:
        """Inject a user text turn."""
        await self._send({"type": "text", "text": text})

    async def send_system_message(self, text: str) -> bool:
        """Push context without asking for a reply; the return reflects provider support."""
        await self._send({"type": "system", "text": text})
        return self.capabilities.system_message

    async def send_tool_result(self, tool_call_id: str, result: str, scheduling: str | None = None) -> None:
        """Return a tool result."""
        await self._send({"type": "tool_result", "tool_call_id": tool_call_id, "result": result, "scheduling": scheduling})

    async def announce_tool_started(self, tool_call_id: str, tool_name: str) -> bool:
        """Signal an async tool started; the return reflects provider support."""
        await self._send({"type": "announce_tool", "tool_call_id": tool_call_id, "name": tool_name})
        return self.capabilities.async_tools

    async def inject_deferred_tool_result(self, tool_call_id: str, tool_name: str, result: str, *, instructions: str) -> None:
        """Complete a deferred tool and prompt one follow-up response."""
        payload = {"type": "inject_tool", "tool_call_id": tool_call_id, "name": tool_name, "result": result, "instructions": instructions}
        await self._send(payload)

    async def update_tools(self, tools: list[dict]) -> bool:
        """Replace the session tools mid-stream; the return reflects provider support."""
        await self._send({"type": "update_tools", "tools": tools})
        return self.capabilities.tool_update

    async def cancel_response(self) -> None:
        """Stop the model's current response (barge-in)."""
        await self._send({"type": "cancel"})

    async def end_input(self) -> None:
        """Tell the server the client finished sending audio."""
        await self._send({"type": "end_input"})

    async def aclose(self) -> None:
        """Close the socket."""
        if self._ws is not None:
            ws, self._ws = self._ws, None
            await ws.close()

    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Iterate the server's ``RealtimeEvent`` stream."""
        return self

    async def __anext__(self) -> RealtimeEvent:
        """Yield the next event; binary frames are audio, text frames are JSON events."""
        if self._ws is None:
            raise StopAsyncIteration
        from websockets.exceptions import ConnectionClosed

        try:
            message = await self._ws.recv()
        except ConnectionClosed:
            raise StopAsyncIteration from None
        if isinstance(message, bytes):
            data = base64.b64encode(message).decode()
            return RealtimeEvent(type=RealtimeEventType.AUDIO, audio=RealtimeAudioData(data=data, sample_rate=self._output_sample_rate))
        return RealtimeEvent.model_validate_json(message)
