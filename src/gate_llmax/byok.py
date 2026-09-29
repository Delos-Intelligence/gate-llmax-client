"""BYOK transport: forward the caller's byok plan on every request as a header."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx

from gate_llmax.models.byok import ByokPlan

BYOK_PLAN_HEADER = "X-Gate-Byok-Plan"


def byok_plan_hook(plan: ByokPlan) -> Callable[[httpx.Request], Awaitable[None]]:
    """A request hook that attaches *plan* as the ``X-Gate-Byok-Plan`` header on every request of a client."""

    async def inject(request: httpx.Request) -> None:
        request.headers[BYOK_PLAN_HEADER] = plan.to_header()

    return inject
