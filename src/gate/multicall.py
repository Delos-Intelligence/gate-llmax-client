"""Parallel multi-model call execution.

Mirrors Hyperion's _process_llm_calls_with_timeout and multi_llm_calls:
- Fan out the same request to N models concurrently via asyncio.gather.
- Apply a global timeout with a 50% grace extension before cancelling.
- Collect all completed results; timed-out tasks produce TIMEOUT responses.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

from gate_common.models.response import GateResponse, RawUsage
from gate_common.types import OutputStatus

logger = logging.getLogger("gate.client.multicall")

T = TypeVar("T")

_DEFAULT_TIMEOUT = 60  # seconds


async def _process_with_timeout(
    tasks: list[asyncio.Task[GateResponse]],
    timeout: float,
) -> None:
    """Wait for all tasks with a global timeout + 50% grace period.

    Mirrors Hyperion's _process_llm_calls_with_timeout:
    1. Wait up to `timeout` seconds.
    2. If exceeded, grant an additional `timeout / 2` seconds.
    3. If still not done, leave tasks to be cancelled by the caller.
    """
    try:
        await asyncio.wait_for(
            asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
            timeout=timeout,
        )
    except TimeoutError:
        grace = timeout / 2
        logger.warning(
            "Global timeout (%.0fs) exceeded granting %.0fs grace period for %d pending tasks",
            timeout,
            grace,
            sum(1 for t in tasks if not t.done()),
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=grace,
            )
        except TimeoutError:
            remaining = [t for t in tasks if not t.done()]
            logger.warning(
                "Grace period exhausted – %d tasks did not complete",
                len(remaining),
            )


async def execute_multicall(
    coros: list[Coroutine[Any, Any, GateResponse]],
    models: list[str],
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[GateResponse]:
    """Execute multiple LLM calls in parallel and return all results.

    Args:
        coros: One coroutine per model (already bound to request + model).
        models: Corresponding model names (used for error responses).
        timeout: Total allowed seconds (+ 50% grace before cancel).

    Returns:
        List of GateResponse, one per model. Failed / timed-out tasks
        produce a GateResponse with status=TIMEOUT or NO_CONNECT.
    """
    tasks: list[asyncio.Task[GateResponse]] = [asyncio.create_task(coro) for coro in coros]

    await _process_with_timeout(tasks, timeout)

    results: list[GateResponse] = []
    for task, model_name in zip(tasks, models, strict=True):
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.warning("Task for model=%s raised: %s", model_name, exc)
                results.append(_timeout_response(model_name))
            else:
                results.append(task.result())
        else:
            task.cancel()
            results.append(_timeout_response(model_name))

    return results


def _timeout_response(model_name: str) -> GateResponse:
    return GateResponse(
        usage=RawUsage(model=model_name, estimated=True),
        model=model_name,
        status=OutputStatus.TIMEOUT,
    )
