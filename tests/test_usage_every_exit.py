"""Every dispatch fires exactly one usage however it ends: Gate's report summed over the legs it reported, else a priced estimate.

Self-contained — a MockTransport stands in for the gateway: it answers chat requests from a script
and list prices from a table, and records the call id of every chat request and every price lookup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import Any
from urllib.parse import unquote
from uuid import UUID

import httpx
import pytest

from gate_llmax import (
    CALL_ID_HEADER,
    LLMClient,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMServerError,
    LLMTimeoutError,
    Message,
    RawUsage,
    StreamChunk,
)
from gate_llmax.client import STREAM_INTERRUPTED
from gate_llmax.fake import FakeClient
from gate_llmax.models.response import ROUND
from gate_llmax.tokens import count_text, estimate_input_tokens

BASE_URL = "http://gate.test"
PROMPT = "ping " * 100
PROMPT_TOKENS = estimate_input_tokens("", [Message.user(PROMPT)], [])
PRICES: dict[str | tuple[str, str], tuple[float, float]] = {"m": (3.0, 15.0), "a": (1.0, 2.0), "b": (5.0, 25.0)}
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}]


def frame(input_tokens: int = 10, output_tokens: int = 5, **extra: Any) -> dict[str, Any]:
    """Gate's own usage frame for a leg: priced, naming the provider."""
    return {
        "is_done": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_tokens / 1000,
        "output_cost": output_tokens / 1000,
        "api_provider": "openai",
        "duration_ms": 100,
        **extra,
    }


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def text(value: str) -> bytes:
    return sse({"text": value})


class Scripted(httpx.AsyncByteStream):
    """Replays ``chunks``, then ends (``[DONE]``), drops the connection, times out, or hangs until cancelled."""

    def __init__(self, chunks: list[bytes], end: str) -> None:
        """Hold the bytes and how the stream ends."""
        self.chunks = chunks
        self.end = end

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Replay the bytes, then end the way the script says."""
        for chunk in self.chunks:
            yield chunk
        if self.end == "drop":
            msg = "connection dropped"
            raise httpx.ReadError(msg)
        if self.end == "timeout":
            msg = "read timed out"
            raise httpx.ReadTimeout(msg)
        if self.end == "hang":
            await asyncio.sleep(3600)
        yield b"data: [DONE]\n\n"


def stream(*chunks: bytes, end: str = "done") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Scripted(list(chunks), end))


def answer(model: str = "m") -> dict[str, Any]:
    usage = {"model": model, "input_tokens": 7, "output_tokens": 2, "input_cost": 0.007, "output_cost": 0.002}
    return {"raw_text": "pong", "model": model, "status": "SUCCESS", "usage": usage}


def model_info(name: str, price: tuple[float, float]) -> dict[str, Any]:
    return {"id": name, "name": name, "capabilities": {}, "input_token_price": price[0], "output_token_price": price[1]}


class Gate:
    """A fake gateway: chat requests answered by ``chat``, ``GET /v1/models/{name}`` from ``prices``."""

    def __init__(self, chat: Callable[[httpx.Request], Any], prices: dict[Any, tuple[float, float]] | None = None) -> None:
        """Hold the chat script and the price table (keyed by name, or by ``(name, plan)``)."""
        self.chat = chat
        self.prices = PRICES if prices is None else prices
        self.ids: list[UUID] = []
        self.lookups: list[tuple[str, str | None]] = []
        self.price_reply: Callable[[httpx.Request], Any] | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        """Answer one request."""
        if request.method == "GET" and request.url.path.startswith("/v1/models/"):
            name = unquote(request.url.path.removeprefix("/v1/models/"))
            plan = request.url.params.get("plan")
            self.lookups.append((name, plan))
            if self.price_reply is not None:
                reply = self.price_reply(request)
                return await reply if asyncio.iscoroutine(reply) else reply
            price = self.prices.get((name, plan)) if plan else None
            price = price or self.prices.get(name)
            if price is None:
                return httpx.Response(404, json={"detail": f"Model '{name}' not found"})
            return httpx.Response(200, json=model_info(name, price))
        self.ids.append(UUID(request.headers[CALL_ID_HEADER]))
        reply = self.chat(request)
        return await reply if asyncio.iscoroutine(reply) else reply


class Usages:
    """A usage callback that keeps what it is handed."""

    def __init__(self) -> None:
        """Start empty."""
        self.seen: list[RawUsage] = []

    async def __call__(self, usage: RawUsage) -> None:
        """Record one fired usage."""
        self.seen.append(usage)

    @property
    def one(self) -> RawUsage:
        """The single usage fired; fails the test if it is not exactly one."""
        assert len(self.seen) == 1, self.seen
        return self.seen[0]


class SlowOnUsage(Usages):
    """An ``on_usage`` writing Gate's usage somewhere slow (a database): it awaits before keeping each one."""

    def __init__(self, *models: str, delay: float = 0.05) -> None:
        """Be slow for ``models`` (every model when none is named), ``delay`` seconds each."""
        super().__init__()
        self.models = models
        self.delay = delay
        self.entered = asyncio.Event()

    async def __call__(self, usage: RawUsage) -> None:
        """Wait on Gate's usage, then keep it; keep an estimate at once."""
        if not usage.estimated and (not self.models or usage.model in self.models):
            self.entered.set()
            await asyncio.sleep(self.delay)
        await super().__call__(usage)


async def cancel_once(entered: asyncio.Event, running: Awaitable[Any]) -> None:
    """Cancel ``running`` once ``entered`` is set, then give the usage it was handing out time to land."""
    task = asyncio.ensure_future(running)
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)


async def drain(chunks: AsyncIterator[Any]) -> None:
    """Read a stream to its end."""
    async for _ in chunks:
        pass


def make_client(gate: Gate, **kwargs: Any) -> tuple[LLMClient, Usages]:
    usages = Usages()
    http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(gate))
    return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http, usage_callback=usages, **kwargs), usages


def scripted(*replies: httpx.Response | Callable[[], Any]) -> Callable[[httpx.Request], Any]:
    """Answer the n-th chat request with the n-th reply (a response, or a callable making one)."""
    queue = list(replies)

    def chat(_request: httpx.Request) -> Any:
        reply = queue.pop(0)
        return reply if isinstance(reply, httpx.Response) else reply()

    return chat


def cost(tokens: int, per_million: float) -> float:
    return round(tokens * per_million / 1_000_000, ROUND)


def assert_estimate(usage: RawUsage, *, input_tokens: int, output_tokens: int, price: tuple[float, float]) -> None:
    assert usage.estimated
    assert (usage.input_tokens, usage.output_tokens) == (input_tokens, output_tokens)
    assert usage.input_cost == cost(input_tokens, price[0])
    assert usage.output_cost == cost(output_tokens, price[1])


async def read_then_close(chunks: AsyncIterator[Any], *, until: Callable[[Any], bool] = lambda _c: True) -> list[Any]:
    """Read until ``until`` accepts a chunk, then stop reading the way a caller does: ``aclose()``."""
    read = []
    async for chunk in chunks:
        read.append(chunk)
        if until(chunk):
            break
    await chunks.aclose()  # ty: ignore[unresolved-attribute]
    return read


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gate_llmax.client.STREAM_RETRY_BACKOFF", 0.0)


# --- a stream that ends ---------------------------------------------------------------------------


async def test_a_completed_stream_fires_gate_s_usage_once() -> None:
    gate = Gate(scripted(stream(text("pong"), sse({"finish_reason": "stop"}), sse(frame()))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    usage = usages.one
    assert not usage.estimated
    assert (usage.input_tokens, usage.output_tokens, usage.total_cost) == (10, 5, 0.015)
    assert usage.call_id == gate.ids[0]
    assert usage.finish_reason == "stop"
    assert gate.lookups == []  # Gate reported, so no price is needed


@pytest.mark.parametrize("usage_chunks", [False, True])
async def test_a_stream_s_usage_keeps_gate_s_reasoning_tokens(usage_chunks: bool) -> None:
    """As a buffered call's does: the end fire and each ``usage_chunks`` fire carry the reasoning tokens Gate reported."""
    gate = Gate(scripted(stream(text("pong"), sse(frame(10, 5, reasoning_tokens=3)))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m", usage_chunks=usage_chunks)]

    assert usages.one.reasoning_tokens == 3


async def test_a_stream_ending_without_usage_fires_a_priced_estimate() -> None:
    gate = Gate(scripted(stream(text("pong"))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("pong"), price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0]
    assert client.total_usage == 0.0  # an estimate is data: a run billing total_usage bills what Gate reported


# --- the caller stops reading ---------------------------------------------------------------------


async def test_a_caller_who_stops_reading_fires_one_estimate() -> None:
    gate = Gate(scripted(stream(sse({"text": "Hello", "reasoning": "hmm"}), text(" world"), end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m"))

    # Fired by aclose() itself, not left to the garbage collector; reasoning delivered is output too.
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Hellohmm"), price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0]


async def test_a_caller_who_stops_after_gate_s_usage_fires_that_usage() -> None:
    gate = Gate(scripted(stream(text("pong"), sse(frame()), end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m"), until=lambda c: c.api_provider is not None)

    assert not usages.one.estimated
    assert (usages.one.input_tokens, usages.one.output_tokens) == (10, 5)


async def test_a_provider_s_unpriced_count_is_not_gate_s_report() -> None:
    """Claude's opening chunk carries the prompt size, unpriced: a stream cut after it is still estimated."""
    gate = Gate(scripted(stream(sse({"input_tokens": 900, "cached_input_tokens": 800}), text("Hi"), end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m"), until=lambda c: c.text == "Hi")

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Hi"), price=PRICES["m"])


async def test_a_streamed_tool_loop_settles_the_turn_its_caller_left() -> None:
    gate = Gate(scripted(stream(text("thinking"), end="hang")))
    client, usages = make_client(gate)

    async def run(_id: str, _name: str, _args: dict[str, Any]) -> str:
        return "found"

    builder = client.request(prompt=PROMPT, operation="t").with_tools(TOOLS, run)
    await read_then_close(builder.call_stream("m"))

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("thinking"), price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0]


async def test_disable_usage_fires_nothing_on_any_exit() -> None:
    gate = Gate(scripted(stream(text("pong"), end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m", disable_usage=True))

    assert usages.seen == []
    assert gate.lookups == []


# --- legs: a resumed stream is one dispatch -------------------------------------------------------


async def test_known_gap_a_resumed_stream_reported_on_its_last_leg_bills_that_leg_alone() -> None:
    """Known gap: the dropped leg's prompt and output are not counted, so the usage is the last leg alone, unflagged.

    Same money as before per-leg metering (the client fired the last usage frame); splitting the reported and
    estimated parts on ``RawUsage`` is tracked separately.
    """
    gate = Gate(scripted(stream(text("Roses are "), end="drop"), stream(text("red."), sse(frame(20, 3)))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert gate.ids[0] == gate.ids[1]
    usage = usages.one
    assert not usage.estimated
    assert (usage.input_tokens, usage.output_tokens, usage.total_cost) == (20, 3, 0.023)
    assert usage.call_id == gate.ids[0]


async def test_a_resumed_stream_reported_on_both_legs_fires_their_sum() -> None:
    gate = Gate(scripted(stream(text("Roses are "), sse(frame(10, 4)), end="drop"), stream(text("red."), sse(frame(20, 3)))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    usage = usages.one
    assert not usage.estimated
    assert (usage.input_tokens, usage.output_tokens, usage.duration_ms) == (30, 7, 200)
    assert usage.total_cost == pytest.approx(0.037)


async def test_a_resumed_stream_never_reported_fires_one_estimate_of_every_leg() -> None:
    gate = Gate(scripted(stream(text("Roses are "), end="drop"), stream(text("red."), sse({"finish_reason": "stop"}))))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    # Each leg is billed its prompt, the second also the text it re-sent, and each what it delivered.
    expected_input = PROMPT_TOKENS + (PROMPT_TOKENS + count_text("Roses are "))
    expected_output = count_text("Roses are ") + count_text("red.")
    assert_estimate(usages.one, input_tokens=expected_input, output_tokens=expected_output, price=PRICES["m"])
    assert usages.one.finish_reason == "stop"


async def test_a_stream_cut_off_after_its_retries_fires_one_interrupted_estimate() -> None:
    gate = Gate(scripted(*[stream(text("partial"), end="drop") for _ in range(3)]))
    client, usages = make_client(gate)

    chunks = [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert chunks[-1].finish_reason == STREAM_INTERRUPTED
    expected_input = 3 * PROMPT_TOKENS + count_text("partial") + count_text("partialpartial")
    assert_estimate(usages.one, input_tokens=expected_input, output_tokens=3 * count_text("partial"), price=PRICES["m"])
    assert usages.one.finish_reason == STREAM_INTERRUPTED


async def test_tool_call_arguments_delivered_count_as_output() -> None:
    call = {"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q": "unf'}}
    gate = Gate(scripted(stream(sse({"tool_calls_delta": [call]}), end="drop")))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text('search{"q": "unf'), price=PRICES["m"])


# --- timeouts, lost connections, errors -----------------------------------------------------------


async def test_a_stream_timing_out_fires_a_priced_estimate_and_raises() -> None:
    gate = Gate(scripted(stream(end="hang")))
    client, usages = make_client(gate, stream_read_timeout=0.05)

    with pytest.raises(LLMTimeoutError) as raised:
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["m"])
    assert usages.one.call_id == raised.value.call_id == gate.ids[0]


def refuse(error: type[httpx.TransportError] = httpx.ConnectError) -> httpx.Response:
    msg = "refused"
    raise error(msg)


def time_out() -> httpx.Response:
    msg = "Gate received the request but has not answered"
    raise httpx.ReadTimeout(msg)


# Each way httpx fails before a request leaves the client whole, and the error the caller gets for it.
UNSENT: dict[type[httpx.TransportError], type[Exception]] = {
    httpx.ConnectError: LLMConnectionError,
    httpx.ConnectTimeout: LLMTimeoutError,
    httpx.PoolTimeout: LLMTimeoutError,
    httpx.WriteTimeout: LLMTimeoutError,
}


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_stream_gate_never_received_fires_nothing(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert usages.seen == []
    assert gate.lookups == []


class WrappedStream(LLMClient):
    """A subclass whose ``_stream`` wraps the client's own, so the builder meters it as one leg."""

    async def _stream(self, request: Any, *, priority: int = 0) -> AsyncIterator[StreamChunk]:
        async for chunk in super()._stream(request, priority=priority):
            yield chunk


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_an_overridden_stream_gate_never_received_fires_nothing(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(partial(refuse, error)))
    usages = Usages()
    http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(gate))
    client = WrappedStream(api_key="k", base_url=BASE_URL, httpx_aclient=http, usage_callback=usages)

    with pytest.raises(UNSENT[error]):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert usages.seen == []
    assert gate.lookups == []


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_resume_that_cannot_connect_fires_the_legs_sent(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(stream(text("Roses are "), end="drop"), partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    # Only the first leg reached Gate: its prompt and what it delivered, nothing for the resume.
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Roses are "), price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0] == gate.ids[1]


@pytest.mark.parametrize(
    ("refusal", "raised"),
    [
        (lambda: httpx.Response(429, json={"detail": "rate limited"}), httpx.HTTPStatusError),
        (lambda: httpx.Response(503, json={"detail": "no route"}), LLMServerError),
        (lambda: stream(sse({"error": {"message": "no", "code": 502}})), LLMServerError),
    ],
    ids=["429", "503", "error-frame"],
)
# The first leg reached Gate whether or not it delivered a chunk before it dropped.
@pytest.mark.parametrize("delivered", ["Roses are ", ""], ids=["after-text", "before-any-chunk"])
async def test_a_refused_resume_fires_the_legs_gate_served(
    refusal: Callable[[], httpx.Response], raised: type[Exception], delivered: str
) -> None:
    first = stream(text(delivered), end="drop") if delivered else stream(end="drop")
    gate = Gate(scripted(first, refusal))
    client, usages = make_client(gate)

    with pytest.raises(raised):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    # Gate refused the resume before any chunk, so served (and billed) nothing for it: the first leg only.
    assert len(gate.ids) == 2
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text(delivered), price=PRICES["m"])


async def test_a_resume_that_cannot_connect_fires_what_gate_reported_for_the_legs_sent() -> None:
    gate = Gate(scripted(stream(text("Roses are "), sse(frame(10, 4)), end="drop"), refuse))
    client, usages = make_client(gate)

    with pytest.raises(LLMConnectionError):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert not usages.one.estimated
    assert (usages.one.input_tokens, usages.one.output_tokens) == (10, 4)


async def test_an_error_after_chunks_fires_and_raises() -> None:
    gate = Gate(scripted(stream(text("half an answer"), sse({"error": {"message": "upstream died", "code": 502}}))))
    client, usages = make_client(gate)

    with pytest.raises(LLMServerError):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("half an answer"), price=PRICES["m"])


@pytest.mark.parametrize(
    "refusal",
    [
        lambda: stream(sse({"error": {"message": "no", "code": 502}})),
        lambda: httpx.Response(404, json={"detail": "no such model"}),
    ],
)
async def test_a_refusal_before_any_chunk_fires_nothing(refusal: Callable[[], httpx.Response]) -> None:
    gate = Gate(scripted(refusal))
    client, usages = make_client(gate)

    with pytest.raises((LLMServerError, LLMModelNotFoundError)):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert usages.seen == []


# --- cancellation ---------------------------------------------------------------------------------


async def test_a_caller_s_timeout_fires_the_usage_then_times_out() -> None:
    gate = Gate(scripted(stream(text("Hello"), end="hang")))
    client, usages = make_client(gate)

    async def read_for_a_while() -> None:
        async with asyncio.timeout(0.1):
            [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    with pytest.raises(TimeoutError):
        await read_for_a_while()

    # "Hello" was delivered before the timeout, so the estimate counts it as output.
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Hello"), price=PRICES["m"])


async def test_a_cancelled_stream_fires_then_stays_cancelled() -> None:
    gate = Gate(scripted(stream(text("Hello"), end="hang")))
    client, usages = make_client(gate)
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in client.request(prompt=PROMPT, operation="t").call_stream("m"):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert usages.one.estimated


async def test_a_second_cancellation_does_not_lose_the_usage() -> None:
    """A cancel scope that keeps cancelling reaches the caller, but the usage still lands."""
    gate = Gate(scripted(stream(text("Hello"), end="hang")))
    billing = asyncio.Event()
    landed: list[RawUsage] = []

    async def slow_bill(usage: RawUsage) -> None:
        billing.set()
        await asyncio.sleep(0.05)
        landed.append(usage)

    client, _ = make_client(gate)
    client.add_usage_callback(slow_bill)
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in client.request(prompt=PROMPT, operation="t").call_stream("m"):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    await billing.wait()
    task.cancel()  # again, while the usage callback runs
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)

    assert len(landed) == 1


# --- buffered calls -------------------------------------------------------------------------------


async def test_a_buffered_call_timing_out_fires_a_priced_estimate() -> None:
    def fail() -> httpx.Response:
        msg = "slow"
        raise httpx.ReadTimeout(msg)

    gate = Gate(scripted(fail))
    client, usages = make_client(gate)

    with pytest.raises(LLMTimeoutError):
        await client.request(prompt=PROMPT, operation="t").call("m")

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0]


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_buffered_call_gate_never_received_fires_nothing(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        await client.request(prompt=PROMPT, operation="t").call("m")

    assert usages.seen == []
    assert gate.lookups == []


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.RemoteProtocolError], ids=lambda error: error.__name__)
async def test_a_buffered_call_losing_its_connection_fires_a_priced_estimate(error: type[httpx.TransportError]) -> None:
    def drop() -> httpx.Response:
        msg = "connection lost before the answer"
        raise error(msg)

    gate = Gate(scripted(drop))
    on_usage = Usages()
    client, usages = make_client(gate)

    with pytest.raises(error):
        await client.request(prompt=PROMPT, operation="t", on_usage=on_usage).call("m")

    # Gate received the request: its prompt is billed as an estimate, under the id Gate logged.
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["m"])
    assert usages.one.call_id == gate.ids[0]
    assert on_usage.one == usages.one


async def test_a_cancelled_buffered_call_fires_an_estimate() -> None:
    async def never() -> httpx.Response:
        await asyncio.sleep(3600)
        raise AssertionError

    gate = Gate(scripted(never))
    on_usage = Usages()
    client, usages = make_client(gate)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await client.request(prompt=PROMPT, operation="t", on_usage=on_usage).call("m")

    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["m"])
    assert on_usage.one == usages.one


def answer_its_model(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=answer(json.loads(request.content)["target"]["model"]))


# Each buffered entry point, a Gate answering it, and the models whose usage Gate reports.
ANSWERED: dict[str, tuple[Callable[[Any], Any], Callable[[httpx.Request], Any], list[str]]] = {
    "call": (lambda builder: builder.call("m"), answer_its_model, ["m"]),
    "multicall": (lambda builder: builder.multicall(["a", "b"]), answer_its_model, ["a", "b"]),
    "batch_multicall": (
        lambda builder: builder.batch_multicall(["a", "b"]),
        lambda _r: httpx.Response(200, json=[answer("a"), answer("b")]),
        ["a", "b"],
    ),
    "server-side call_best": (
        lambda builder: builder.call_best(["a", "b"], lowest="price"),
        lambda _r: httpx.Response(200, json=answer("a")),
        ["a"],
    ),
}


@pytest.mark.parametrize("entry", ANSWERED)
async def test_a_call_cancelled_while_handing_out_gate_s_usage_still_fires_it(entry: str) -> None:
    send, chat, models = ANSWERED[entry]
    gate = Gate(chat)
    on_usage = SlowOnUsage()
    client, usages = make_client(gate)

    await cancel_once(on_usage.entered, send(client.request(prompt=PROMPT, operation="t", on_usage=on_usage)))

    # Gate answered and billed: its usage reaches every callback once, the caller's cancel notwithstanding.
    assert [(u.model, u.estimated) for u in usages.seen] == [(model, False) for model in models]
    assert {u.call_id for u in usages.seen} == set(gate.ids)
    assert on_usage.seen == usages.seen


async def test_a_batch_cancelled_while_pricing_a_timed_out_answer_still_fires_every_answer() -> None:
    pricing = asyncio.Event()

    async def slow_price(_request: httpx.Request) -> httpx.Response:
        pricing.set()
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=model_info("b", PRICES["b"]))

    timed_out = {"model": "b", "status": "TIMEOUT", "usage": {"model": "b", "estimated": True}}
    gate = Gate(lambda _r: httpx.Response(200, json=[answer("a"), timed_out]))
    gate.price_reply = slow_price
    client, usages = make_client(gate)

    await cancel_once(pricing, client.request(prompt=PROMPT, operation="t").batch_multicall(["a", "b"]))

    assert [(u.model, u.estimated) for u in usages.seen] == [("a", False), ("b", True)]
    assert_estimate(usages.seen[1], input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])


async def test_a_buffered_refusal_fires_nothing() -> None:
    gate = Gate(scripted(httpx.Response(503, json={"detail": "down"})))
    client, usages = make_client(gate)

    with pytest.raises(LLMServerError):
        await client.request(prompt=PROMPT, operation="t").call("m")

    assert usages.seen == []


async def test_call_prefer_bills_each_attempt_as_a_plain_call_would() -> None:
    def timeout() -> httpx.Response:
        msg = "slow"
        raise httpx.ReadTimeout(msg)

    attempts = (
        httpx.Response(404, json={"detail": "gone"}),
        httpx.Response(503, json={"detail": "down"}),
        refuse,
        timeout,
        httpx.Response(200, json=answer("m")),
    )
    gate = Gate(scripted(*attempts))
    on_usage = Usages()
    client, usages = make_client(gate)

    response = await client.request(prompt=PROMPT, operation="t", on_usage=on_usage).call_prefer(["a", "a", "a", "b", "m"])

    assert response.raw_text == "pong"
    _gone, _down, _unreached, slow, good = gate.ids
    # A refusal or an unreached Gate bills nothing; the timeout bills one estimate, at its model's price.
    assert [u.call_id for u in usages.seen] == [slow, good]
    assert_estimate(usages.seen[0], input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])
    assert not usages.seen[1].estimated
    assert on_usage.seen == usages.seen
    assert gate.lookups == [("b", None)]


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_multicall_prices_a_timed_out_model_and_bills_nothing_for_a_refused_one(error: type[httpx.TransportError]) -> None:
    async def reply(request: httpx.Request) -> Any:
        model = json.loads(request.content)["target"]["model"]
        if model == "b":
            await asyncio.sleep(5)
        if model == "gone":
            return httpx.Response(404, json={"detail": "gone"})
        if model == "unreached":
            return refuse(error)
        return httpx.Response(200, json=answer(model))

    gate = Gate(reply)
    on_usage = Usages()
    client, usages = make_client(gate)

    models = ["a", "b", "gone", "unreached"]
    responses = await client.request(prompt=PROMPT, operation="t", on_usage=on_usage).multicall(models, timeout=0.05)

    assert [r.status for r in responses[1:]] == ["TIMEOUT"] * 3  # statuses unchanged; only the bill differs
    assert_estimate(responses[1].usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])
    assert {r.usage.call_id for r in responses} == set(gate.ids)
    assert on_usage.seen == usages.seen == [responses[0].usage, responses[1].usage]
    assert gate.lookups == [("b", None)]


async def test_batch_multicall_prices_gate_s_timed_out_answer() -> None:
    timed_out = {"model": "b", "status": "TIMEOUT", "usage": {"model": "b", "estimated": True}}
    gate = Gate(lambda _r: httpx.Response(200, json=[answer("a"), timed_out]))
    client, usages = make_client(gate)

    responses = await client.request(prompt=PROMPT, operation="t").batch_multicall(["a", "b"])

    assert_estimate(responses[1].usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])
    assert [u.call_id for u in usages.seen] == [gate.ids[0], gate.ids[0]]


async def a_wins_b_hangs(request: httpx.Request) -> httpx.Response:
    model = json.loads(request.content)["target"]["model"]
    if model == "b":
        await asyncio.sleep(3600)
    return httpx.Response(200, json=answer(model))


async def test_call_best_bills_its_cancelled_loser_before_it_returns() -> None:
    gate = Gate(a_wins_b_hangs)
    client, usages = make_client(gate)

    response = await client.request(prompt=PROMPT, operation="t").call_best(["a", "b"])

    assert response.model == "a"
    assert [(u.model, u.estimated) for u in usages.seen] == [("a", False), ("b", True)]
    assert_estimate(usages.seen[1], input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])
    assert {u.call_id for u in usages.seen} == set(gate.ids)


def test_call_best_sync_bills_its_cancelled_loser() -> None:
    # The loop ends as call_best returns, cancelling whatever is still pending.
    gate = Gate(a_wins_b_hangs)
    client, usages = make_client(gate)

    response = client.request(prompt=PROMPT, operation="t").call_best_sync(["a", "b"])

    assert response.model == "a"
    assert [(u.model, u.estimated) for u in usages.seen] == [("a", False), ("b", True)]


async def test_call_best_bills_gate_s_usage_for_a_loser_cancelled_while_handing_it_out() -> None:
    async def b_answers_first(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["target"]["model"] == "a":
            await asyncio.sleep(0.01)
        return answer_its_model(request)

    gate = Gate(b_answers_first)
    on_usage = SlowOnUsage("b", delay=0.1)  # still writing b's usage when a wins and b is cancelled
    client, usages = make_client(gate)

    response = await client.request(prompt=PROMPT, operation="t", on_usage=on_usage).call_best(["a", "b"])

    assert response.model == "a"
    assert sorted((u.model, u.estimated) for u in usages.seen) == [("a", False), ("b", False)]
    assert {u.call_id for u in usages.seen} == set(gate.ids)


# --- server-side fan-outs: one request, every model under its call id -----------------------------

# Each fan-out sent as one Gate request, and the models it bills an estimate for when Gate never
# answers: call_best's one answer, priced as its first model (the winner is unknown); one per model for a batch.
FAN_OUTS: dict[str, tuple[Callable[[Any], Any], list[str]]] = {
    "call_best": (lambda builder: builder.call_best(["a", "b"], lowest="price"), ["a"]),
    "batch_multicall": (lambda builder: builder.batch_multicall(["a", "b"]), ["a", "b"]),
}


@pytest.mark.parametrize("fan_out", FAN_OUTS)
async def test_a_fan_out_timing_out_fires_a_priced_estimate_per_model_billed(fan_out: str) -> None:
    send, billed = FAN_OUTS[fan_out]
    gate = Gate(scripted(time_out))
    on_usage = Usages()
    client, usages = make_client(gate)

    with pytest.raises(LLMTimeoutError) as raised:
        await send(client.request(prompt=PROMPT, operation="t", on_usage=on_usage))

    assert [u.model for u in usages.seen] == billed
    for usage in usages.seen:
        assert_estimate(usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES[usage.model])
        assert usage.call_id == raised.value.call_id == gate.ids[0]
    assert on_usage.seen == usages.seen


@pytest.mark.parametrize("fan_out", FAN_OUTS)
async def test_a_cancelled_fan_out_fires_a_priced_estimate_per_model_billed(fan_out: str) -> None:
    async def never() -> httpx.Response:
        await asyncio.sleep(3600)
        raise AssertionError

    send, billed = FAN_OUTS[fan_out]
    gate = Gate(scripted(never))
    client, usages = make_client(gate)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await send(client.request(prompt=PROMPT, operation="t"))

    assert [u.model for u in usages.seen] == billed
    for usage in usages.seen:
        assert_estimate(usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES[usage.model])
        assert usage.call_id == gate.ids[0]


@pytest.mark.parametrize("fan_out", FAN_OUTS)
@pytest.mark.parametrize("error", [httpx.ReadError, httpx.RemoteProtocolError], ids=lambda error: error.__name__)
async def test_a_fan_out_losing_its_connection_fires_a_priced_estimate_per_model_billed(
    fan_out: str, error: type[httpx.TransportError]
) -> None:
    def drop() -> httpx.Response:
        msg = "connection lost before the answer"
        raise error(msg)

    send, billed = FAN_OUTS[fan_out]
    gate = Gate(scripted(drop))
    on_usage = Usages()
    client, usages = make_client(gate)

    with pytest.raises(error):
        await send(client.request(prompt=PROMPT, operation="t", on_usage=on_usage))

    # Gate received the request, as a plain call() losing its connection: each model it ran bills its prompt.
    assert [u.model for u in usages.seen] == billed
    for usage in usages.seen:
        assert_estimate(usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES[usage.model])
        assert usage.call_id == gate.ids[0]
    assert on_usage.seen == usages.seen


@pytest.mark.parametrize("fan_out", FAN_OUTS)
@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_fan_out_gate_never_received_fires_nothing(fan_out: str, error: type[httpx.TransportError]) -> None:
    send, _billed = FAN_OUTS[fan_out]
    gate = Gate(scripted(partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        await send(client.request(prompt=PROMPT, operation="t"))

    assert usages.seen == []
    assert gate.lookups == []


@pytest.mark.parametrize("fan_out", FAN_OUTS)
@pytest.mark.parametrize("status", [404, 503])
async def test_a_refused_fan_out_fires_nothing(fan_out: str, status: int) -> None:
    send, _billed = FAN_OUTS[fan_out]
    gate = Gate(scripted(httpx.Response(status, json={"detail": "no"})))
    client, usages = make_client(gate)

    with pytest.raises((LLMModelNotFoundError, LLMServerError)):
        await send(client.request(prompt=PROMPT, operation="t"))

    assert usages.seen == []


def batch_frame(index: int, model: str) -> bytes:
    """One model's answer on a streamed fan-out, with the usage Gate reported for it."""
    return sse({"index": index, "model": model, "response": answer(model)})


async def test_a_multicall_stream_its_caller_leaves_estimates_every_model_not_yet_answered() -> None:
    gate = Gate(scripted(stream(batch_frame(1, "b"), end="hang")))
    on_usage = Usages()
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t", on_usage=on_usage).multicall_stream(["a", "b", "m"]))

    # b answered with Gate's usage; a and m were still running at Gate: a priced estimate each, under the one call id.
    assert [(u.model, u.estimated) for u in usages.seen] == [("b", False), ("a", True), ("m", True)]
    for usage in usages.seen[1:]:
        assert_estimate(usage, input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES[usage.model])
    assert {u.call_id for u in usages.seen} == {gate.ids[0]}
    assert on_usage.seen == usages.seen


async def test_a_cancelled_multicall_stream_estimates_every_model_not_yet_answered() -> None:
    gate = Gate(scripted(stream(batch_frame(0, "a"), end="hang")))
    client, usages = make_client(gate)
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"]):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert [(u.model, u.estimated) for u in usages.seen] == [("a", False), ("b", True)]
    assert_estimate(usages.seen[1], input_tokens=PROMPT_TOKENS, output_tokens=0, price=PRICES["b"])


async def test_a_multicall_stream_cancelled_while_handing_out_an_answer_s_usage_still_fires_it() -> None:
    gate = Gate(scripted(stream(batch_frame(1, "b"), end="hang")))
    on_usage = SlowOnUsage()
    client, usages = make_client(gate)

    await cancel_once(on_usage.entered, drain(client.request(prompt=PROMPT, operation="t", on_usage=on_usage).multicall_stream(["a", "b"])))

    # b answered: Gate's usage, landing on its own; a was still running at Gate: its estimate.
    assert sorted((u.model, u.estimated) for u in usages.seen) == [("a", True), ("b", False)]


async def test_a_multicall_stream_cut_by_its_caller_s_timeout_before_any_answer_estimates_every_model() -> None:
    gate = Gate(scripted(stream(end="hang")))
    client, usages = make_client(gate)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert [(u.model, u.estimated) for u in usages.seen] == [("a", True), ("b", True)]


async def test_a_multicall_stream_timing_out_estimates_every_model_not_yet_answered() -> None:
    gate = Gate(scripted(stream(batch_frame(0, "a"), end="timeout")))
    client, usages = make_client(gate)

    with pytest.raises(LLMTimeoutError):
        [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert [(u.model, u.estimated) for u in usages.seen] == [("a", False), ("b", True)]
    assert usages.seen[1].call_id == gate.ids[0]


async def test_a_multicall_stream_losing_its_connection_estimates_every_model_not_yet_answered() -> None:
    gate = Gate(scripted(stream(batch_frame(1, "b"), end="drop")))
    client, usages = make_client(gate)

    with pytest.raises(httpx.ReadError):
        [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b", "m"])]

    assert [(u.model, u.estimated) for u in usages.seen] == [("b", False), ("a", True), ("m", True)]
    assert {u.call_id for u in usages.seen} == {gate.ids[0]}


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.RemoteProtocolError], ids=lambda error: error.__name__)
async def test_a_multicall_stream_losing_its_connection_before_any_answer_estimates_every_model(
    error: type[httpx.TransportError],
) -> None:
    def drop() -> httpx.Response:
        msg = "connection lost before the answer"
        raise error(msg)

    gate = Gate(scripted(drop))
    client, usages = make_client(gate)

    with pytest.raises(error):
        [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert [(u.model, u.estimated) for u in usages.seen] == [("a", True), ("b", True)]
    assert {u.call_id for u in usages.seen} == {gate.ids[0]}


async def test_a_multicall_stream_failing_after_an_answer_estimates_every_model_not_yet_answered() -> None:
    class Failing(Usages):
        async def __call__(self, usage: RawUsage) -> None:
            await super().__call__(usage)
            if not usage.estimated:
                msg = "the caller's callback failed"
                raise RuntimeError(msg)

    gate = Gate(scripted(stream(batch_frame(1, "b"), end="hang")))
    on_usage = Failing()
    client, usages = make_client(gate)

    with pytest.raises(RuntimeError):
        [f async for f in client.request(prompt=PROMPT, operation="t", on_usage=on_usage).multicall_stream(["a", "b", "m"])]

    # Gate answered b and was still running a and m: an estimate each, as a stream erroring after chunks bills.
    assert [(u.model, u.estimated) for u in on_usage.seen] == [("b", False), ("a", True), ("m", True)]
    assert [(u.model, u.estimated) for u in usages.seen if u.estimated] == [("a", True), ("m", True)]


async def test_a_multicall_stream_that_ends_fires_each_answer_once() -> None:
    gate = Gate(scripted(stream(batch_frame(1, "b"), batch_frame(0, "a"))))
    client, usages = make_client(gate)

    [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert [(u.model, u.estimated) for u in usages.seen] == [("b", False), ("a", False)]


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_multicall_stream_gate_never_received_fires_nothing(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert usages.seen == []
    assert gate.lookups == []


@pytest.mark.parametrize("status", [404, 503])
async def test_a_refused_multicall_stream_fires_nothing(status: int) -> None:
    gate = Gate(scripted(httpx.Response(status, json={"detail": "no"})))
    client, usages = make_client(gate)

    with pytest.raises((LLMModelNotFoundError, LLMServerError)):
        [f async for f in client.request(prompt=PROMPT, operation="t").multicall_stream(["a", "b"])]

    assert usages.seen == []


# --- the speech stream: Gate bills the text's characters, and sends no usage frame ----------------

SPEECH = "Hello world"


def speech(*chunks: bytes, end: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=Scripted(list(chunks), end))


def assert_speech_estimate(usage: RawUsage) -> None:
    assert_estimate(usage, input_tokens=len(SPEECH), output_tokens=0, price=PRICES["m"])


async def test_a_completed_speech_stream_fires_a_priced_estimate_of_its_text() -> None:
    gate = Gate(scripted(httpx.Response(200, content=b"ID3\xff\xfb")))
    client, usages = make_client(gate)
    responses: list[Any] = []

    async def keep(response: Any) -> None:
        responses.append(response)

    audio = [c async for c in client.audio("speech", text=SPEECH, operation="t").callback(keep).call_stream("m")]

    assert audio == [b"ID3\xff\xfb"]
    assert_speech_estimate(usages.one)
    assert usages.one.call_id == gate.ids[0]
    assert [r.usage for r in responses] == [usages.one]  # the response callbacks still get the finished stream


async def test_a_speech_stream_its_caller_leaves_fires_one_estimate() -> None:
    gate = Gate(scripted(speech(b"ID3", end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.audio("speech", text=SPEECH, operation="t").call_stream("m"))

    assert_speech_estimate(usages.one)
    assert usages.one.call_id == gate.ids[0]


async def test_a_cancelled_speech_stream_fires_one_estimate() -> None:
    gate = Gate(scripted(speech(b"ID3", end="hang")))
    client, usages = make_client(gate)
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in client.audio("speech", text=SPEECH, operation="t").call_stream("m"):
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert_speech_estimate(usages.one)


@pytest.mark.parametrize("reply", [time_out, lambda: speech(b"ID3", end="timeout")], ids=["before-the-audio", "after-some-audio"])
async def test_a_speech_stream_timing_out_fires_one_estimate_and_raises(reply: Callable[[], httpx.Response]) -> None:
    gate = Gate(scripted(reply))
    client, usages = make_client(gate)

    with pytest.raises(LLMTimeoutError) as raised:
        [c async for c in client.audio("speech", text=SPEECH, operation="t").call_stream("m")]

    assert_speech_estimate(usages.one)
    assert usages.one.call_id == raised.value.call_id == gate.ids[0]


async def test_a_speech_stream_failing_after_audio_fires_one_estimate_and_raises() -> None:
    gate = Gate(scripted(speech(b"ID3", end="drop")))
    client, usages = make_client(gate)

    with pytest.raises(httpx.ReadError):
        [c async for c in client.audio("speech", text=SPEECH, operation="t").call_stream("m")]

    assert_speech_estimate(usages.one)


@pytest.mark.parametrize("error", UNSENT, ids=lambda error: error.__name__)
async def test_a_speech_stream_gate_never_received_fires_nothing(error: type[httpx.TransportError]) -> None:
    gate = Gate(scripted(partial(refuse, error)))
    client, usages = make_client(gate)

    with pytest.raises(UNSENT[error]):
        [c async for c in client.audio("speech", text=SPEECH, operation="t").call_stream("m")]

    assert usages.seen == []
    assert gate.lookups == []


@pytest.mark.parametrize("status", [404, 503])
async def test_a_refused_speech_stream_fires_nothing(status: int) -> None:
    gate = Gate(scripted(httpx.Response(status, json={"detail": "no"})))
    client, usages = make_client(gate)

    with pytest.raises((LLMModelNotFoundError, LLMServerError)):
        [c async for c in client.audio("speech", text=SPEECH, operation="t").call_stream("m")]

    assert usages.seen == []


# --- the list price -------------------------------------------------------------------------------


async def test_a_group_is_priced_as_the_model_it_serves_on_the_plan() -> None:
    prices = {"group_mezzo": (9.0, 9.0), ("group_mezzo", "omicron"): (2.0, 8.0)}
    gate = Gate(scripted(stream(text("Hello"), end="hang"), stream(text("Hello"), end="hang")), prices=prices)
    client, usages = make_client(gate, default_plan="omicron")

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("group_mezzo"))
    view = client.with_usage_callback(Usages())
    await read_then_close(view.request(prompt=PROMPT, operation="t").call_stream("group_mezzo"))

    assert gate.lookups == [("group_mezzo", "omicron")]  # fetched once, then cached for the client and its views
    for usage in usages.seen:
        assert usage.model == "group_mezzo"
        assert_estimate(usage, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Hello"), price=(2.0, 8.0))


async def test_the_model_that_served_sets_the_price() -> None:
    gate = Gate(scripted(stream(sse({"text": "Hi", "final_model": "b"}), end="hang")))
    client, usages = make_client(gate)

    await read_then_close(client.request(prompt=PROMPT, operation="t").stream().call_prefer(["a", "b"]))

    assert gate.lookups == [("b", None)]
    assert usages.one.model == "a"
    assert_estimate(usages.one, input_tokens=PROMPT_TOKENS, output_tokens=count_text("Hi"), price=PRICES["b"])


async def test_an_expired_price_is_fetched_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gate_llmax.metering.PRICE_TTL", 0.0)
    gate = Gate(scripted(stream(text("a")), stream(text("b"))))
    client, _ = make_client(gate)

    for _ in range(2):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert gate.lookups == [("m", None), ("m", None)]


async def test_without_a_price_the_estimate_fires_at_cost_0_and_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    gate = Gate(scripted(stream(text("pong")), stream(text("pong"))))
    gate.price_reply = lambda _r: httpx.Response(500, json={"detail": "catalog down"})
    client, usages = make_client(gate)

    with caplog.at_level(logging.WARNING, logger="gate.client.metering"):
        for _ in range(2):
            [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert len(usages.seen) == 2
    for usage in usages.seen:
        assert usage.estimated
        assert (usage.input_tokens, usage.output_tokens, usage.total_cost) == (PROMPT_TOKENS, count_text("pong"), 0.0)
    assert len([r for r in caplog.records if "No list price" in r.message]) == 1
    assert len(gate.lookups) == 1  # the failure is remembered, so the second exit does not wait on it


async def test_a_slow_price_does_not_hold_the_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gate_llmax.metering.PRICE_WAIT", 0.05)
    release = asyncio.Event()

    async def slow(_request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=model_info("m", PRICES["m"]))

    gate = Gate(scripted(stream(text("pong"))))
    gate.price_reply = slow
    client, usages = make_client(gate)

    async with asyncio.timeout(1):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]

    assert usages.one.estimated
    assert usages.one.total_cost == 0.0
    release.set()
    await asyncio.sleep(0.01)  # the lookup finishes in the background, for the next exit
    assert await client._list_price("m", None) == PRICES["m"]


# --- usage_chunks=True and the fake client --------------------------------------------------------


async def test_usage_chunks_adds_an_estimate_only_when_gate_reported_nothing() -> None:
    gate = Gate(scripted(stream(text("pong"), sse(frame())), stream(text("pong"), end="hang")))
    client, usages = make_client(gate)

    [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m", usage_chunks=True)]
    assert [u.estimated for u in usages.seen] == [False]

    await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m", usage_chunks=True))
    assert [u.estimated for u in usages.seen] == [False, True]


async def test_usage_chunks_cancelled_while_handing_out_gate_s_usage_still_fires_it() -> None:
    gate = Gate(scripted(stream(text("pong"), sse(frame()), end="hang")))
    on_usage = SlowOnUsage()
    client, usages = make_client(gate)

    builder = client.request(prompt=PROMPT, operation="t", on_usage=on_usage)
    await cancel_once(on_usage.entered, drain(builder.call_stream("m", usage_chunks=True)))

    assert [(u.estimated, u.input_tokens, u.output_tokens) for u in usages.seen] == [(False, 10, 5)]


async def test_the_fake_client_reports_its_streams_and_prices_nothing(caplog: pytest.LogCaptureFixture) -> None:
    usages = Usages()
    client = FakeClient("one two three", usage_callback=usages)

    with caplog.at_level(logging.WARNING, logger="gate.client.metering"):
        [c async for c in client.request(prompt=PROMPT, operation="t").call_stream("m")]
        await read_then_close(client.request(prompt=PROMPT, operation="t").call_stream("m"))

    assert [u.estimated for u in usages.seen] == [False, True]
    assert all(u.total_cost == 0.0 for u in usages.seen)
    assert not [r for r in caplog.records if "No list price" in r.message]
