"""The client's ceilings are the OUTER rungs of the Gate timeout ladder.

Gate's per-attempt ceiling is a model's ``timeout`` plus its 30s watchdog margin
(``GATEWAY_MAX_BUDGET``). Every client-side ceiling must sit strictly above it, so a
timeout is raised — and named — by the gateway, which knows what upstream did. When the
client fires first, all the caller learns is that something, somewhere, took too long.
"""

from __future__ import annotations

import httpx

from gate_llmax.client import (
    CLIENT_MARGIN,
    CONNECT_TIMEOUT,
    DEFAULT_TIMEOUT,
    GATE_MAX_TRIES,
    GATEWAY_MAX_BUDGET,
    MEDIA_CLIENT_TIMEOUT,
    MEDIA_MAX_TRIES,
    STREAM_READ_TIMEOUT,
    LLMClient,
    client_ceiling,
)


def test_client_margin_is_positive() -> None:
    """A zero margin puts the client and the gateway on the same tick."""
    assert CLIENT_MARGIN > 0


def test_buffered_ceiling_is_above_the_gateway() -> None:
    assert DEFAULT_TIMEOUT > GATEWAY_MAX_BUDGET


def test_stream_ceiling_is_above_the_gateway() -> None:
    assert STREAM_READ_TIMEOUT > GATEWAY_MAX_BUDGET


def test_media_ceiling_is_above_the_whole_retry_ladder() -> None:
    """Gate may re-run the whole per-attempt budget on another deployment; a one-attempt ceiling cuts the retry off."""
    assert MEDIA_CLIENT_TIMEOUT > GATEWAY_MAX_BUDGET * MEDIA_MAX_TRIES


def test_buffered_ceiling_covers_a_retried_call() -> None:
    assert DEFAULT_TIMEOUT > GATEWAY_MAX_BUDGET * GATE_MAX_TRIES


def test_a_request_timeout_raises_the_client_ceiling() -> None:
    """The bug this guards: a 1800s video request still cut at the default ceiling."""
    assert client_ceiling(1800, 2) > 1800 * 2


def test_connect_is_far_below_every_read_budget() -> None:
    """A gateway that is down should fail fast, not sit on a 10-minute read budget."""
    assert min(DEFAULT_TIMEOUT, STREAM_READ_TIMEOUT) > CONNECT_TIMEOUT


def test_a_shared_transport_cannot_shorten_the_client_ceiling() -> None:
    """The bug class this guards: a caller's pool default silently capping Gate."""
    shared = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    client = LLMClient(api_key="k", base_url="http://gate", httpx_aclient=shared)
    assert client._timeout == DEFAULT_TIMEOUT
    assert client._stream_timeout.read == STREAM_READ_TIMEOUT


def test_explicit_ceilings_are_honoured() -> None:
    client = LLMClient(api_key="k", base_url="http://gate", timeout=700.0, stream_read_timeout=800.0)
    assert client._timeout == 700.0
    assert client._stream_timeout.read == 800.0
