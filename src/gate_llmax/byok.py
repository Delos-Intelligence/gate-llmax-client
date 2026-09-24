"""BYOK transport: bind a caller's upstream credential to the send in flight and forward it as a header."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from gate_llmax.models.byok import ByokCredential, ByokPlan

BYOK_HEADER = "X-Gate-Upstream-Credentials"
BYOK_PLAN_HEADER = "X-Gate-Byok-Plan"

# The caller's own upstream credential for the send in flight; the httpx request hook turns it into a
# header. Request-scoped, never on the request body, so it never reaches the gateway's usage log.
provider_key_var: ContextVar[ByokCredential | None] = ContextVar("provider_key_var", default=None)


@contextmanager
def provider_key_scope(credential: ByokCredential | None) -> Iterator[None]:
    """Bind the caller's provider key for the duration of a send, resetting it however the send exits."""
    if credential is None:
        yield
        return
    token = provider_key_var.set(credential)
    try:
        yield
    finally:
        provider_key_var.reset(token)


async def inject_byok_header(request: httpx.Request) -> None:
    """Request hook: attach the scoped BYOK credential as a header when one is bound."""
    credential = provider_key_var.get()
    if credential is not None:
        request.headers[BYOK_HEADER] = credential.to_header()


def byok_plan_hook(plan: ByokPlan) -> Callable[[httpx.Request], Awaitable[None]]:
    """A request hook that attaches *plan* as the ``X-Gate-Byok-Plan`` header on every request of a client."""

    async def inject(request: httpx.Request) -> None:
        request.headers[BYOK_PLAN_HEADER] = plan.to_header()

    return inject
