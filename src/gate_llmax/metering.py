"""What a dispatch cost when Gate could not say: a stream's legs, list prices, and the estimate built from them.

Gate reports a streamed dispatch's usage on one frame per leg (a leg is one HTTP request; a
dropped connection resumes in a new one). A stream the caller stops reading, cancels, or loses
before that frame has no report, so the client estimates it: the prompt each leg was sent and
the output each leg delivered, at the model's list price, flagged ``estimated``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Self
from uuid import UUID

import httpx

from .exceptions import LLMConnectionError, LLMTimeoutError
from .models.response import ROUND, RawUsage, StreamChunk
from .tokens import count_text

logger = logging.getLogger("gate.client.metering")

PRICE_WAIT = 2.0  # longest an exit waits for a list price; past it the estimate fires at cost 0
PRICE_TTL = 300.0  # how long a list price is reused
PRICE_RETRY = 60.0  # how long a failed lookup stands before an exit asks Gate again
_CHARS_PER_TOKEN = 4
# httpx timeouts raised before a request leaves the client (connecting, waiting for a pooled
# connection): like a refused connection, Gate never received it.
UNSENT_TIMEOUTS = (httpx.ConnectTimeout, httpx.PoolTimeout)

type PriceKey = tuple[str, str | None]


class ListPrice(NamedTuple):
    """A model's list price in USD per 1M tokens, as Gate's catalog states it."""

    input: float
    output: float


class PriceCache:
    """List prices by ``(model, plan)``, shared by a client and its views. A lookup is bounded and never raises."""

    def __init__(self) -> None:
        """Start empty."""
        self._known: dict[PriceKey, tuple[float, ListPrice | None]] = {}
        self._loading: dict[PriceKey, asyncio.Task[ListPrice | None]] = {}
        self._warned: set[PriceKey] = set()

    async def get(self, model: str, plan: str | None, fetch: Callable[[str, str | None], Awaitable[ListPrice]]) -> ListPrice | None:
        """``model``'s list price on ``plan``, or ``None`` when it can't be had within ``PRICE_WAIT``."""
        key = (model, plan)
        known = self._known.get(key)
        if known is not None and time.monotonic() < known[0]:
            return known[1]
        loop = asyncio.get_running_loop()
        task = self._loading.get(key)
        if task is None or task.done() or task.get_loop() is not loop:
            task = loop.create_task(self._load(key, fetch))
            self._loading[key] = task
        try:
            async with asyncio.timeout(PRICE_WAIT):
                # Shielded: a waiter giving up must not cancel the lookup the next exit reuses.
                return await asyncio.shield(task)
        except TimeoutError:
            self._warn(key, f"no answer within {PRICE_WAIT:.0f}s")
            return None

    async def _load(self, key: PriceKey, fetch: Callable[[str, str | None], Awaitable[ListPrice]]) -> ListPrice | None:
        price: ListPrice | None = None
        try:
            price = await fetch(*key)
        except Exception as exc:  # an estimate fires whatever the lookup did
            self._warn(key, repr(exc))
        self._known[key] = (time.monotonic() + (PRICE_TTL if price is not None else PRICE_RETRY), price)
        if self._loading.get(key) is asyncio.current_task():
            del self._loading[key]
        return price

    def _warn(self, key: PriceKey, why: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("No list price for model=%s plan=%s (%s); its estimates fire at cost 0", key[0], key[1], why)


def reports_usage(chunk: StreamChunk) -> bool:
    """Whether ``chunk`` is Gate's usage frame for its leg: Gate names the provider and prices the call there.

    A provider's own token counts can ride an earlier chunk unpriced (Claude's opening chunk carries
    the prompt size); that is not a report, so a leg cut off after it is still estimated.
    """
    return chunk.api_provider is not None or chunk.input_cost is not None or chunk.output_cost is not None


def bears_usage(chunk: StreamChunk) -> bool:
    """Whether ``chunk`` carries any usage at all, Gate's frame or a provider's counts — what ``usage_chunks=True`` fires on."""
    return reports_usage(chunk) or chunk.input_tokens is not None or chunk.output_tokens is not None


def chunk_usage(chunk: StreamChunk, *, model: str, operation: str, call_id: UUID) -> RawUsage:
    """The usage one chunk reports."""
    return RawUsage(
        input_tokens=chunk.input_tokens or 0,
        output_tokens=chunk.output_tokens or 0,
        cached_input_tokens=chunk.cached_input_tokens or 0,
        input_cost=chunk.input_cost or 0.0,
        output_cost=chunk.output_cost or 0.0,
        cached_input_cost=chunk.cached_input_cost or 0.0,
        model=model,
        api_provider=chunk.api_provider or "",
        hosting_provider=chunk.hosting_provider or "",
        region=chunk.region or "",
        duration_ms=chunk.duration_ms or 0,
        ttft_ms=chunk.ttft_ms,
        operation=operation,
        call_id=call_id,
    )


def estimated_usage(
    price: ListPrice | None,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    operation: str,
    call_id: UUID | None,
    finish_reason: str = "",
) -> RawUsage:
    """A usage Gate never reported, priced at ``price`` (cost 0 without one) and flagged ``estimated``."""
    return RawUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=round(input_tokens * price.input / 1_000_000, ROUND) if price is not None else 0.0,
        output_cost=round(output_tokens * price.output / 1_000_000, ROUND) if price is not None else 0.0,
        estimated=True,
        operation=operation,
        call_id=call_id,
        finish_reason=finish_reason,
    )


def timed_out_unsent(error: BaseException) -> bool:
    """Whether ``error`` is an ``LLMTimeoutError`` raised from a timeout before its request left the client: it owes nothing."""
    return isinstance(error, LLMTimeoutError) and isinstance(error.__cause__, UNSENT_TIMEOUTS)


def _tokens(text: str) -> int:
    try:
        return count_text(text)
    except Exception:  # a missing encoder must not cost the estimate
        return len(text) // _CHARS_PER_TOKEN


@dataclass
class Leg:
    """One HTTP request of a streamed dispatch: the first, or a resume after a dropped connection."""

    prefix: str  # text delivered before this leg, sent back to the model as an assistant turn to continue
    output: list[str] = field(default_factory=list)  # what this leg delivered: text, reasoning, tool-call deltas
    report: StreamChunk | None = None  # Gate's usage frame for this leg; cumulative, so a later one replaces it


class StreamMeter:
    """What one streamed dispatch was sent, delivered and reported, leg by leg."""

    def __init__(self) -> None:
        """Start before the first leg."""
        self.legs: list[Leg] = []
        self.chunks = 0
        self.served_model: str | None = None
        self.finish_reason = ""
        self.native_finish_reason = ""

    def start_leg(self, prefix: str = "") -> None:
        """A request goes out, re-sending ``prefix`` (the text already delivered) when it resumes."""
        self.legs.append(Leg(prefix=prefix))

    def unsent(self) -> None:
        """The current leg could not connect: Gate never received it, so it is no leg at all."""
        self.legs.pop()

    def observe(self, chunk: StreamChunk) -> None:
        """Record one chunk of the current leg on its way to the caller."""
        leg = self.legs[-1]
        self.chunks += 1
        leg.output.extend(part for part in (chunk.text, chunk.reasoning) if part)
        for delta in chunk.tool_calls_delta or []:
            function = delta.get("function")
            if isinstance(function, dict):
                leg.output.extend(part for part in (function.get("name"), function.get("arguments")) if isinstance(part, str) and part)
        self.served_model = chunk.final_model or self.served_model
        self.finish_reason = chunk.finish_reason or self.finish_reason
        self.native_finish_reason = chunk.native_finish_reason or self.native_finish_reason
        if reports_usage(chunk):
            leg.report = chunk

    @property
    def reported(self) -> bool:
        """Whether Gate reported any leg."""
        return any(leg.report is not None for leg in self.legs)

    def owes_usage(self, error: BaseException | None) -> bool:
        """Whether a dispatch leaving with ``error`` (``None`` for its end) owes a usage.

        It does once a request reached Gate, unless Gate refused it before any chunk: a refusal
        fires nothing, while a timeout, a cancellation or a caller who stopped reading leave a
        request that ran. A leg that could not connect was dropped (``unsent``), so a dispatch Gate
        never received owes nothing, and one whose resume could not connect owes its earlier legs.
        """
        if not self.legs:
            return False
        if error is None or isinstance(error, (GeneratorExit, asyncio.CancelledError, LLMTimeoutError, LLMConnectionError)):
            return True
        return self.chunks > 0

    def reported_usage(self, *, model: str, operation: str, call_id: UUID) -> RawUsage:
        """The legs Gate reported, summed; a leg it never reported adds nothing (Gate's log still has it)."""
        usages = [chunk_usage(leg.report, model=model, operation=operation, call_id=call_id) for leg in self.legs if leg.report is not None]
        total = usages[0]
        for usage in usages[1:]:
            total += usage
        return total.model_copy(update={"finish_reason": self.finish_reason, "native_finish_reason": self.native_finish_reason})

    def estimated_tokens(self, prompt_tokens: int) -> tuple[int, int]:
        """``(input, output)`` as if Gate billed each leg: its prompt plus the text it re-sent, and what it delivered."""
        input_tokens = sum(prompt_tokens + _tokens(leg.prefix) for leg in self.legs)
        output_tokens = sum(_tokens("".join(leg.output)) for leg in self.legs)
        return input_tokens, output_tokens


class MeteredStream:
    """A streamed dispatch's chunks, with the ``StreamMeter`` following them leg by leg."""

    def __init__(self, chunks: AsyncIterator[StreamChunk], meter: StreamMeter) -> None:
        """Wrap ``chunks``, which keep ``meter`` up to date as they are read."""
        self._chunks = chunks
        self.meter = meter

    @classmethod
    def of(cls, stream: AsyncIterator[StreamChunk]) -> MeteredStream:
        """``stream`` itself when it meters its legs (the client's), else one leg metered here (a stub, an override)."""
        if isinstance(stream, MeteredStream):
            return stream
        meter = StreamMeter()
        return cls(_one_leg(stream, meter), meter)

    def __aiter__(self) -> Self:
        """Iterate the chunks."""
        return self

    async def __anext__(self) -> StreamChunk:
        """The next chunk."""
        return await self._chunks.__anext__()

    async def aclose(self) -> None:
        """Close the stream underneath, ending its HTTP request."""
        await aclose(self._chunks)


async def _one_leg(stream: AsyncIterator[StreamChunk], meter: StreamMeter) -> AsyncIterator[StreamChunk]:
    meter.start_leg()
    try:
        async for chunk in stream:
            meter.observe(chunk)
            yield chunk
    except LLMConnectionError:
        meter.unsent()
        raise
    finally:
        await aclose(stream)


async def aclose(stream: object) -> None:
    """Close ``stream`` now when it can be (an async generator, a ``MeteredStream``) rather than at garbage collection."""
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


class Closing[T]:
    """``async with Closing(stream) as chunks:`` closes ``stream`` on the way out, abandoned or cancelled too.

    So the dispatch behind it settles (its usage fires, its connection closes) when its reader
    leaves, not when the garbage collector finalizes it.
    """

    def __init__(self, stream: AsyncIterator[T]) -> None:
        """Hold ``stream``."""
        self._stream = stream

    async def __aenter__(self) -> AsyncIterator[T]:
        """The stream itself."""
        return self._stream

    async def __aexit__(self, *_exc: object) -> None:
        """Close it."""
        await aclose(self._stream)


_SETTLING: set[asyncio.Task[None]] = set()


async def settle(fire: Coroutine[Any, Any, None], *, quiet: bool) -> None:
    """Run an exit's usage ``fire`` to its end even if the caller is cancelled meanwhile.

    It runs as its own task: a cancellation reaching the caller (its ``asyncio.timeout``, a cancel
    scope cancelling every await) still propagates, while the fire finishes on its own. ``quiet``
    logs a failing callback instead of raising it, for an exit that already carries an exception
    a billing error must not replace.
    """
    task = asyncio.ensure_future(fire)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        _SETTLING.add(task)
        task.add_done_callback(_settled)
        raise
    except Exception:
        if not quiet:
            raise
        logger.exception("A usage callback failed on a dispatch's way out")


def _settled(task: asyncio.Task[None]) -> None:
    _SETTLING.discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.error("A usage callback failed after its caller was cancelled", exc_info=exc)
