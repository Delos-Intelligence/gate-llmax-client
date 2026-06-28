"""Client-side rate limiting, built into ``LLMClient`` (no wrapper class needed).

A single-process throttle over several dimensions; an empty ``RateLimit()`` is a no-op
(no semaphore, no waiting), so the client can always go through the limiter and skip
branching on whether one is configured:

- ``max_concurrency``     — at most N calls in flight at once.
- ``requests_per_minute`` — minimum spacing between successive calls.
- ``tokens_per_minute``   — after each call, push the next allowed start out by the tokens it spent.
- per-call ``priority``   — when calls queue for a concurrency slot, higher priority is served first.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic import BaseModel


class RateLimit(BaseModel):
    """Per-client rate-limit config — pass to ``LLMClient(rate_limit=...)``. Every field unset = unlimited."""

    max_concurrency: int | None = None
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None


class RateLimiter:
    """Enforces a :class:`RateLimit`. With every field unset it is a no-op."""

    def __init__(self, limit: RateLimit) -> None:
        """Build the limiter from a :class:`RateLimit` config."""
        self._max_concurrency = limit.max_concurrency
        self._min_interval = 60.0 / limit.requests_per_minute if limit.requests_per_minute else 0.0
        self._tokens_per_minute = limit.tokens_per_minute
        self._active = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []  # heap of (-priority, seq, future)
        self._seq = 0
        self._next_allowed = 0.0
        self._interval_lock = asyncio.Lock()

    @asynccontextmanager
    async def guard(self, priority: int = 0) -> AsyncIterator[None]:
        """Acquire a concurrency slot (highest ``priority`` first when queued), wait the interval, then yield."""
        await self._acquire(priority)
        try:
            await self._wait_turn()
            yield
        finally:
            self._release()

    async def _acquire(self, priority: int) -> None:
        # asyncio is cooperative: the state mutations below are atomic (no await until ``await fut``).
        if self._max_concurrency is None:
            return
        if self._active < self._max_concurrency:
            self._active += 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (-priority, self._seq, fut))
        self._seq += 1
        await fut

    def _release(self) -> None:
        if self._max_concurrency is None:
            return
        # Hand the freed slot to the highest-priority waiter (active count unchanged), else free it.
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._active -= 1

    async def _wait_turn(self) -> None:
        if not self._min_interval and not self._tokens_per_minute:
            return
        async with self._interval_lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = self._next_allowed
            self._next_allowed = now + self._min_interval

    def record_tokens(self, tokens: int) -> None:
        """Push the next allowed start out by what the last call's token budget implies."""
        if self._tokens_per_minute and tokens > 0:
            self._next_allowed += 60.0 * tokens / self._tokens_per_minute
