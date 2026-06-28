"""Client-side rate limiting, built into ``LLMClient`` (no wrapper class needed).

A single-process throttle over three dimensions; any field left ``None`` is unlimited:

- ``max_concurrency``    — at most N calls in flight at once (a semaphore).
- ``requests_per_minute``— minimum spacing between successive calls.
- ``tokens_per_minute``  — after each call, push the next allowed start out by the token budget it spent.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic import BaseModel


class RateLimit(BaseModel):
    """Per-client rate-limit configuration. Pass to ``LLMClient(rate_limit=...)``."""

    max_concurrency: int | None = None
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None


class RateLimiter:
    """Enforces a :class:`RateLimit`: a concurrency semaphore + a request/token interval gate."""

    def __init__(self, limit: RateLimit) -> None:
        """Build the limiter from a :class:`RateLimit` config."""
        self._sem = asyncio.Semaphore(limit.max_concurrency) if limit.max_concurrency else None
        self._min_interval = 60.0 / limit.requests_per_minute if limit.requests_per_minute else 0.0
        self._tokens_per_minute = limit.tokens_per_minute
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """Hold a concurrency slot and wait for the request interval, then yield."""
        if self._sem is not None:
            await self._sem.acquire()
        try:
            await self._wait_turn()
            yield
        finally:
            if self._sem is not None:
                self._sem.release()

    async def _wait_turn(self) -> None:
        async with self._lock:
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
