"""Call ids: every Gate request carries one UUIDv7 in ``X-Gate-Call-Id``, and the usage that reports it carries the same.

Self-contained — a MockTransport stands in for the gateway and records the header of every leg.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest

from gate_llmax import CALL_ID_HEADER, LLMBudgetError, LLMClient, LLMConnectionError, LLMServerError, RawUsage, new_call_id
from gate_llmax import ids as ids_mod
from gate_llmax.ids import uuid7_fallback
from gate_llmax.models.request import LLMRequest, SingleTarget

BASE_URL = "http://gate.test"
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}]


class Gate:
    """A fake gateway: answers with ``reply(request)`` and records the call id and body of every request."""

    def __init__(self, reply: Callable[[httpx.Request], Any]) -> None:
        """Hold the per-request reply function."""
        self.reply = reply
        self.ids: list[UUID] = []
        self.bodies: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the request, then answer it (``reply`` may be sync or async, and may raise)."""
        self.ids.append(UUID(request.headers[CALL_ID_HEADER]))
        self.bodies.append(json.loads(request.content))
        response = self.reply(request)
        return await response if asyncio.iscoroutine(response) else response


class Usages:
    """A usage callback that keeps what it is handed."""

    def __init__(self) -> None:
        """Start empty."""
        self.seen: list[RawUsage] = []

    async def __call__(self, usage: RawUsage) -> None:
        """Record one fired usage."""
        self.seen.append(usage)

    @property
    def ids(self) -> list[UUID | None]:
        """The call id of every usage fired, in order."""
        return [usage.call_id for usage in self.seen]


def make_client(gate: Gate, **kwargs: Any) -> LLMClient:
    http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(gate))
    return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http, **kwargs)


def model_of(request: httpx.Request) -> str:
    return json.loads(request.content)["target"]["model"]


def answer(model: str = "m", *, tool_calls: list[dict] | None = None, status: str = "SUCCESS") -> dict[str, Any]:
    return {"raw_text": "pong", "model": model, "status": status, "tool_calls": tool_calls, "usage": {"model": model, "input_tokens": 3}}


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def event_stream(*chunks: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join([*chunks, b"data: [DONE]\n\n"]))


class DroppingStream(httpx.AsyncByteStream):
    """Replays ``chunks``, then dies the way a severed connection does."""

    def __init__(self, chunks: list[bytes]) -> None:
        """Hold the bytes to replay before the connection dies."""
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Replay the canned bytes, then raise the transport error httpx would."""
        for chunk in self.chunks:
            yield chunk
        msg = "connection dropped"
        raise httpx.ReadError(msg)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def unix_ms(call_id: UUID) -> int:
    """The UUIDv7 timestamp: its first 48 bits."""
    return call_id.int >> 80


def assert_uuid7(call_id: UUID | None, *, between: tuple[int, int]) -> None:
    assert call_id is not None
    assert call_id.version == 7
    assert call_id.variant == uuid.RFC_4122
    assert between[0] <= unix_ms(call_id) <= between[1]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gate_llmax.client.STREAM_RETRY_BACKOFF", 0.0)


# --- the wire: one header per request, never a body field --------------------------------------


async def test_a_buffered_call_sends_one_uuid7_and_its_usage_carries_it() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=answer()))
    usages, on_usage = Usages(), Usages()
    client = make_client(gate, usage_callback=usages)

    before = now_ms()
    response = await client.request(prompt="ping", operation="t", on_usage=on_usage).call("m")
    after = now_ms()

    [call_id] = gate.ids
    assert_uuid7(call_id, between=(before, after))
    assert response.usage.call_id == call_id  # Gate's JSON has none: the client stamps the id it sent
    assert usages.ids == on_usage.ids == [call_id]
    assert "call_id" not in gate.bodies[0]  # the header is the only wire channel


async def test_a_stream_sends_one_uuid7_and_its_usage_carries_it() -> None:
    usage_frame = {"text": "ng", "is_done": True, "input_tokens": 3, "output_tokens": 1, "input_cost": 0.0, "api_provider": "p"}
    gate = Gate(lambda _r: event_stream(sse({"text": "po"}), sse(usage_frame)))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    before = now_ms()
    text = "".join([c.text async for c in client.request(prompt="ping", operation="t").call_stream("m")])
    after = now_ms()

    assert text == "pong"
    [call_id] = gate.ids
    assert_uuid7(call_id, between=(before, after))
    assert usages.ids == [call_id]
    assert not usages.seen[0].estimated


async def test_every_resumed_leg_of_a_stream_sends_the_same_id() -> None:
    def reply(_request: httpx.Request) -> httpx.Response:
        if len(gate.ids) < 3:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DroppingStream([sse({"text": "a"})]))
        return event_stream(sse({"text": "b", "is_done": True, "input_tokens": 1, "output_tokens": 1}))

    gate = Gate(reply)
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    text = "".join([c.text async for c in client.request(prompt="ping", operation="t").call_stream("m")])

    assert text == "aab"
    assert len(gate.ids) == 3  # the first leg plus two resumes
    assert len(set(gate.ids)) == 1  # one Gate request, however many legs
    assert gate.bodies[2]["messages"][-1]["role"] == "assistant"  # really a resume, not a replay
    assert usages.ids == [gate.ids[0]]


@pytest.mark.parametrize(
    ("build", "path"),
    [
        (lambda c: c.embed("hi", operation="t"), "/v1/embeddings"),
        (lambda c: c.image("a cat", operation="t"), "/v1/images"),
        (lambda c: c.audio("speech", text="hi", operation="t"), "/v1/audio/speech"),
        (lambda c: c.audio("music", prompt="hum", operation="t"), "/v1/audio/generations"),
        (lambda c: c.video("a wave", operation="t"), "/v1/videos"),
        (lambda c: c.vision(["aGk="], operation="t"), "/v1/vision/ocr"),
        (lambda c: c.transcribe("YQ==", operation="t"), "/v1/audio/transcriptions"),
        (lambda c: c.isolation("YQ==", operation="t"), "/v1/audio/isolation"),
        (lambda c: c.dub("https://x.test/a.mp4", "en", "fr", operation="t"), "/v1/audio/dubbing"),
    ],
)
async def test_a_media_call_sends_the_id_and_its_usage_carries_it(build: Callable[[LLMClient], Any], path: str) -> None:
    gate = Gate(lambda r: httpx.Response(200, json={"model": "x", "usage": {"model": "x"}}) if r.url.path == path else httpx.Response(418))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    before = now_ms()
    response = await build(client).call("x")

    [call_id] = gate.ids
    assert_uuid7(call_id, between=(before, now_ms()))
    assert response.usage.call_id == call_id
    assert usages.ids == [call_id]
    assert "call_id" not in gate.bodies[0]


async def test_an_image_stream_sends_the_id_and_its_terminal_usage_carries_it() -> None:
    gate = Gate(lambda _r: event_stream(sse({"b64": "aGk="}), sse({"usage": {"model": "x", "output_tokens": 9}})))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    frames = [frame async for frame in client.image("a cat", operation="t").stream().call("x")]

    assert len(frames) == 1
    assert usages.ids == gate.ids
    assert usages.seen[0].output_tokens == 9


async def test_a_tts_stream_sends_the_id_and_its_usage_carries_it() -> None:
    gate = Gate(lambda _r: httpx.Response(200, content=b"\x00\x01"))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    audio = [chunk async for chunk in client.audio("speech", text="hi", operation="t").call_stream("x")]

    assert audio == [b"\x00\x01"]
    assert len(gate.ids) == 1
    assert usages.ids == gate.ids


async def test_a_parallel_target_is_one_request_so_its_answers_share_the_id() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=[answer("a"), answer("b", status="TIMEOUT")]))
    usages, on_usage = Usages(), Usages()
    client = make_client(gate, usage_callback=usages)

    responses = await client.request(prompt="ping", operation="t", on_usage=on_usage).batch_multicall(["a", "b"])

    [call_id] = gate.ids
    assert [r.usage.call_id for r in responses] == [call_id, call_id]
    assert on_usage.ids == [call_id, call_id]
    assert on_usage.seen[1].estimated  # Gate's timed-out answer is estimated under the batch's id too
    assert usages.ids == [call_id, call_id]


async def test_a_request_sent_directly_keeps_its_own_id_or_gets_one() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=answer()))
    client = make_client(gate)
    own = new_call_id()

    await client._send(LLMRequest(target=SingleTarget(model="m"), call_id=own))
    await client._send(LLMRequest(target=SingleTarget(model="m")))

    assert gate.ids[0] == own
    assert gate.ids[1] != own
    assert gate.ids[1].version == 7


async def test_a_direct_stream_without_an_id_names_all_its_legs_with_one() -> None:
    def reply(_request: httpx.Request) -> httpx.Response:
        if len(gate.ids) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DroppingStream([sse({"text": "a"})]))
        return event_stream(sse({"text": "b", "is_done": True}))

    gate = Gate(reply)
    client = make_client(gate)

    [c async for c in client._stream(LLMRequest(target=SingleTarget(model="m"), stream=True))]

    assert len(gate.ids) == 2
    assert len(set(gate.ids)) == 1


# --- one id per Gate request ---------------------------------------------------------------------


async def test_each_tool_loop_turn_is_its_own_request() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=answer(tool_calls=[TOOL_CALL] if len(gate.ids) == 1 else None)))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    async def run(_id: str, _name: str, _args: dict[str, Any]) -> str:
        return "found"

    response = await client.request(prompt="look it up", operation="t").with_tools(TOOLS, run).call("m")

    assert len(gate.ids) == 2
    assert gate.ids[0] != gate.ids[1]
    assert usages.ids == gate.ids  # each turn bills under its own request
    assert response.usage.call_id == gate.ids[0]  # the summed usage keeps the first id (RawUsage.__add__)


async def test_each_streamed_tool_loop_turn_is_its_own_request() -> None:
    def reply(_request: httpx.Request) -> httpx.Response:
        if len(gate.ids) == 1:
            return event_stream(sse({"tool_calls_delta": [{"index": 0, **TOOL_CALL}]}), sse({"is_done": True, "input_tokens": 1}))
        return event_stream(sse({"text": "done", "is_done": True, "input_tokens": 1}))

    gate = Gate(reply)
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    async def run(_id: str, _name: str, _args: dict[str, Any]) -> str:
        return "found"

    own = new_call_id()
    builder = client.request(prompt="look it up", operation="t").with_tools(TOOLS, run).with_call_id(own)
    [c async for c in builder.call_stream("m")]

    assert gate.ids[0] == own  # the caller's id names the first turn...
    assert gate.ids[1] != own  # ...and only that one
    assert usages.ids == gate.ids


async def test_multicall_sends_one_request_per_model_and_estimates_a_timeout_under_its_id() -> None:
    async def reply(request: httpx.Request) -> httpx.Response:
        if model_of(request) == "slow":
            await asyncio.sleep(5)
        return httpx.Response(200, json=answer(model_of(request)))

    gate = Gate(reply)
    on_usage = Usages()
    client = make_client(gate)

    responses = await client.request(prompt="ping", operation="t", on_usage=on_usage).multicall(["fast", "slow"], timeout=0.05)

    sent = dict(zip([body["target"]["model"] for body in gate.bodies], gate.ids, strict=True))
    assert sent["fast"] != sent["slow"]
    assert [r.usage.call_id for r in responses] == [sent["fast"], sent["slow"]]
    assert on_usage.ids == [sent["fast"], sent["slow"]]
    assert on_usage.seen[1].estimated


def timed_out(_request: httpx.Request) -> httpx.Response:
    msg = "slow"
    raise httpx.ReadTimeout(msg)


async def test_call_prefer_estimates_a_failed_attempt_under_that_attempt_s_id() -> None:
    gate = Gate(lambda r: timed_out(r) if model_of(r) == "bad" else httpx.Response(200, json=answer()))
    on_usage = Usages()
    client = make_client(gate)

    response = await client.request(prompt="ping", operation="t", on_usage=on_usage).call_prefer(["bad", "good"])

    bad, good = gate.ids
    assert bad != good
    assert on_usage.ids == [bad, good]
    assert on_usage.seen[0].estimated
    assert response.usage.call_id == good


async def test_call_prefer_through_a_tool_loop_estimates_the_turn_that_failed() -> None:
    """The failed attempt is the turn that raised, not the first one, which already billed under its own id."""

    def reply(request: httpx.Request) -> httpx.Response:
        if model_of(request) == "b":
            return httpx.Response(200, json=answer("b"))
        if len(gate.ids) == 1:
            return httpx.Response(200, json=answer("a", tool_calls=[TOOL_CALL]))
        return timed_out(request)

    gate = Gate(reply)
    on_usage = Usages()
    client = make_client(gate)

    async def run(_id: str, _name: str, _args: dict[str, Any]) -> str:
        return "found"

    await client.request(prompt="ping", operation="t", on_usage=on_usage).with_tools(TOOLS, run).call_prefer(["a", "b"])

    first_turn, failed_turn, fallback = gate.ids
    assert on_usage.ids == [first_turn, failed_turn, fallback]
    assert on_usage.seen[1].estimated


# --- the caller's own id -------------------------------------------------------------------------


async def test_a_caller_s_id_names_the_next_dispatch_only() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=answer()))
    client = make_client(gate)
    own = new_call_id()

    builder = client.request(prompt="ping", operation="t").with_call_id(own)
    first = await builder.call("m")
    second = await builder.call("m")

    assert gate.ids[0] == own
    assert first.usage.call_id == own
    assert gate.ids[1] != own
    assert second.usage.call_id == gate.ids[1]


async def test_a_caller_s_id_survives_a_cast_and_a_stream_view_once() -> None:
    gate = Gate(lambda r: event_stream(sse({"text": "x"})) if json.loads(r.content)["stream"] else httpx.Response(200, json=answer()))
    client = make_client(gate)
    cast_id, stream_id = new_call_id(), new_call_id()

    builder = client.request(prompt="ping", operation="t")
    await builder.with_call_id(cast_id).cast_json().call("m")
    await builder.call("m")
    view = builder.with_call_id(stream_id).stream()
    [c async for c in view.call_prefer(["m", "n"])]
    [c async for c in view.call_prefer(["m", "n"])]

    assert gate.ids[0] == cast_id
    assert gate.ids[1] not in {cast_id, stream_id}
    assert gate.ids[2] == stream_id
    assert gate.ids[3] not in {cast_id, stream_id}


async def test_a_media_builder_takes_a_caller_s_id_once() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json={"model": "x", "usage": {"model": "x"}}))
    client = make_client(gate)
    own = new_call_id()

    builder = client.embed("hi", operation="t").with_call_id(own)
    await builder.call("x")
    await builder.call("x")

    assert gate.ids[0] == own
    assert gate.ids[1] != own


# --- estimate paths and errors -------------------------------------------------------------------


async def test_a_stream_that_cannot_connect_raises_under_its_id_and_bills_nothing() -> None:
    def reply(_request: httpx.Request) -> httpx.Response:
        msg = "refused"
        raise httpx.ConnectError(msg)

    gate = Gate(reply)
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    with pytest.raises(LLMConnectionError) as raised:
        [c async for c in client.request(prompt="ping", operation="t").call_stream("m")]

    [call_id] = gate.ids
    assert raised.value.call_id == call_id
    assert usages.ids == []  # Gate never received it


async def test_a_stream_ending_without_usage_is_estimated_under_its_id() -> None:
    gate = Gate(lambda _r: event_stream(sse({"text": "pong"})))
    usages = Usages()
    client = make_client(gate, usage_callback=usages)

    [c async for c in client.request(prompt="ping", operation="t").call_stream("m")]

    assert usages.ids == gate.ids
    assert usages.seen[0].estimated


async def test_a_failed_buffered_call_names_its_request() -> None:
    gate = Gate(lambda _r: httpx.Response(503, json={"detail": "down"}))
    client = make_client(gate)

    with pytest.raises(LLMServerError) as raised:
        await client.request(prompt="ping", operation="t").call("m")

    assert raised.value.call_id == gate.ids[0]


async def test_a_budget_denial_names_no_request() -> None:
    gate = Gate(lambda _r: httpx.Response(200, json=answer()))
    client = make_client(gate)

    async def deny() -> bool:
        return False

    with pytest.raises(LLMBudgetError) as raised:
        await client.request(prompt="ping", operation="t").budget(deny).call("m")

    assert raised.value.call_id is None
    assert gate.ids == []


def test_summed_usage_keeps_the_first_id() -> None:
    a, b = new_call_id(), new_call_id()
    assert (RawUsage(call_id=a) + RawUsage(call_id=b)).call_id == a
    assert (RawUsage() + RawUsage(call_id=b)).call_id == b
    assert (RawUsage() + RawUsage()).call_id is None


# --- realtime ------------------------------------------------------------------------------------


class FakeSocket:
    """A websocket that replays ``frames`` and records what the client sends."""

    def __init__(self, frames: list[str]) -> None:
        """Hold the frames the server will send."""
        self.frames = frames
        self.sent: list[str] = []

    async def recv(self) -> str:
        """The next server frame."""
        return self.frames.pop(0)

    async def send(self, data: str) -> None:
        """Record a client frame."""
        self.sent.append(data)

    async def close(self) -> None:
        """Nothing to close."""


@pytest.mark.parametrize("own", [None, new_call_id()])
async def test_a_realtime_session_names_its_handshake_and_its_usage(monkeypatch: pytest.MonkeyPatch, own: UUID | None) -> None:
    handshakes: list[dict[str, str]] = []
    socket = FakeSocket([json.dumps({"capabilities": {}}), json.dumps({"type": "session_usage", "session_usage": {"model": "rt"}})])

    async def connect(_url: str, *, additional_headers: dict[str, str]) -> FakeSocket:
        handshakes.append(additional_headers)
        return socket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    usages = Usages()
    client = LLMClient(api_key="k", base_url=BASE_URL, usage_callback=usages)

    async with client.realtime("rt", **({"call_id": own} if own else {})) as session:
        await session.__anext__()

    [headers] = handshakes
    assert session.call_id is not None
    assert headers[CALL_ID_HEADER] == str(session.call_id)
    assert own is None or session.call_id == own
    assert usages.ids == [session.call_id]
    assert "call_id" not in json.loads(socket.sent[0])["session"]


# --- minting ---------------------------------------------------------------------------------------


def test_the_fallback_mints_a_valid_uuid7_on_any_python() -> None:
    before = now_ms()
    minted = [uuid7_fallback() for _ in range(1000)]
    after = now_ms()

    for call_id in minted:
        assert_uuid7(call_id, between=(before, after))
    assert len(set(minted)) == len(minted)


def test_new_call_id_falls_back_before_python_3_14(monkeypatch: pytest.MonkeyPatch) -> None:
    used: list[UUID] = []

    def spy() -> UUID:
        used.append(uuid7_fallback())
        return used[-1]

    monkeypatch.setattr(ids_mod, "sys", SimpleNamespace(version_info=(3, 12, 0)))
    monkeypatch.setattr(ids_mod, "uuid7_fallback", spy)

    call_id = new_call_id()

    assert used == [call_id]
    assert call_id.version == 7
