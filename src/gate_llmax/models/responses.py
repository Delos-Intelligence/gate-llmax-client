"""Models for the OpenAI Responses API passthrough — an escape hatch.

This endpoint forwards a raw Responses call to the configured upstream deployment.
It bypasses Gate's typed request normalization and exists only for features the
unified chat surface does not cover (e.g. the OpenAI Responses tool runtime).
Prefer ``/v1/chat/completions``.
"""

from __future__ import annotations

from pydantic import Field

from ..types import JsonDict, JsonValue
from .request import CallControl
from .response import GateCallRecord


class ResponsesRequest(CallControl):
    """Raw OpenAI Responses request. Only ``input`` and ``extra_body`` are forwarded.

    The upstream endpoint and credentials are taken from the server-side deployment
    selected by ``model`` — never from the caller — so this cannot target an
    arbitrary host or leak provider keys. ``extra_body`` is merged into the request
    JSON body (not into SDK call kwargs), so it cannot inject transport-level options.
    """

    model: str
    input: JsonValue = Field(description="Responses `input`: a string or a list of input items.")
    extra_body: JsonDict | None = Field(
        default=None,
        description="Additional Responses parameters (tools, reasoning, instructions, …) merged into the request body.",
    )
    operation: str = Field(default="", description="Caller-supplied usage tag echoed onto the usage log.")


class ResponsesResponse(GateCallRecord):
    """Response from ``/v1/responses`` — Gate call metadata + the raw Responses object."""

    output: JsonValue | None = Field(default=None, description="The upstream Responses object, serialized as JSON.")
