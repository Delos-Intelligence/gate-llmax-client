"""RequestBuilder, CastedRequestBuilder, and TypedGateResponse for the Gate client."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gate_llmax.models.audio_gen import AudioGenRequest
from gate_llmax.models.images import ImageData, ImageRequest, ImageResponse
from gate_llmax.models.messages import Message, TextMessage
from gate_llmax.models.request import BestTarget, GateRequest, ParallelTarget, RequestSpecifics, SingleTarget, ZoneSelection
from gate_llmax.models.response import BaseAudioResponse, GateCallRecord, GateResponse, RawUsage, StreamChunk, ToolCall
from gate_llmax.models.tts import TTSRequest, TTSResponse
from gate_llmax.models.video import VideoRequest, VideoResponse
from gate_llmax.types import JsonDict, OutputStatus

from .exceptions import (
    GateBudgetError,
    GateConnectionError,
    GateModelNotFoundError,
    GateServerError,
    GateTimeoutError,
)
from .multicall import execute_multicall
from .parsing import extract_json_from_text
from .tokens import count, estimate_input_tokens

if TYPE_CHECKING:
    from .client import GateClient

logger = logging.getLogger("gate.client.request")

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MULTICALL_TIMEOUT = 60.0

OnUsage = Callable[[RawUsage], Awaitable[None]]
UsageCallback = Callable[[RawUsage], Awaitable[None]]
"""Endpoint-agnostic callback. Fires once per response with that response's `RawUsage`."""

GateCallback = Callable[[GateResponse], Awaitable[None]]
ImageCallback = Callable[[ImageResponse], Awaitable[None]]
AudioCallback = Callable[[BaseAudioResponse], Awaitable[None]]
VideoCallback = Callable[[VideoResponse], Awaitable[None]]

ToolExecutor = Callable[[str, str, dict[str, Any]], Awaitable[str]]
"""Runs one tool call: ``(tool_call_id, name, parsed_arguments) -> result_text``."""


class ToolProgress(BaseModel):
    """Intermediate text from a streaming tool, surfaced live into the output stream."""

    content: str


class ToolResult(BaseModel):
    """Terminal item from a streaming tool: the result fed back to the model.

    ``output`` becomes the tool message content. ``redo`` controls whether the model is
    re-invoked after the tools run — ``False`` ends the loop (e.g. the tool already
    produced the final answer and there is nothing left for the model to do).
    """

    output: str | None = None
    redo: bool = True


ToolStreamItem = ToolProgress | ToolResult
StreamingToolExecutor = Callable[[str, str, dict[str, Any]], AsyncIterator[ToolStreamItem]]
"""Streams one tool call: ``(tool_call_id, name, parsed_args) -> AsyncIterator[ToolStreamItem]``.

Yield ``ToolProgress`` items to surface live progress into the output stream, and one
``ToolResult`` to feed the result back to the model (``redo=False`` ends the loop)."""

BudgetCheck = Callable[[], Awaitable[bool]]
"""Pre-call budget gate: returns True to allow the call, False to deny it with `GateBudgetError`."""

MAX_TOOL_ITERS = 8

_TOOL_BUDGET_EXCEEDED_NOTE = (
    "You can no longer call any tools due to the size of the context. With the current information, answer the user's question directly."
)


def _conversation_token_count(messages: list[Message]) -> int:
    """Exact token count of all text content across ``messages`` (for the tool-budget guard)."""
    return sum(count(block.text) for msg in messages for block in msg.content if isinstance(block, TextMessage))


def _extract_tool_calls(response: GateResponse) -> tuple[str, list[ToolCall] | None]:
    """Pull (assistant_text, tool_calls) out of a ``GateResponse``."""
    return response.raw_text, response.tool_calls


def _accumulate_tool_call_deltas(acc: dict[int, ToolCall], deltas: list[Any]) -> None:
    """Fold streaming ``tool_calls_delta`` fragments into ``acc`` keyed by index."""
    for raw in deltas:
        delta = raw if isinstance(raw, dict) else raw.model_dump()
        idx = delta.get("index") or 0
        slot = acc.setdefault(idx, ToolCall())
        if delta.get("id"):
            slot.id = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            slot.function.name = fn["name"]
        if fn.get("arguments"):
            slot.function.arguments += fn["arguments"]


async def _run_tool_call(tool_call: ToolCall, executor: ToolExecutor) -> tuple[ToolCall, str]:
    """Parse one tool call's arguments and run it through ``executor``."""
    raw_args = tool_call.function.arguments or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    output = await executor(tool_call.id, tool_call.function.name, args)
    return tool_call, output


async def _drain_streaming_tool(
    executor: StreamingToolExecutor,
    tool_call: ToolCall,
    queue: asyncio.Queue[tuple[ToolCall, ToolStreamItem | None]],
) -> None:
    """Run one streaming tool, pushing each item to ``queue``; a trailing ``None`` signals done."""
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    async for item in executor(tool_call.id, tool_call.function.name, args):
        await queue.put((tool_call, item))
    await queue.put((tool_call, None))


def first_ready_success[R](results: dict[int, R], is_success: Callable[[R], bool]) -> tuple[int, R] | None:
    """Earliest index whose contiguous-from-0 prefix is fully resolved and holds a success."""
    i = 0
    while i in results:
        if is_success(results[i]):
            return i, results[i]
        i += 1
    return None


async def select_best[I, R](
    items: list[I],
    runner: Callable[[I], Coroutine[Any, Any, R]],
    *,
    is_success: Callable[[R], bool],
    score: Callable[[R], float] | None = None,
    accept: Callable[[R], bool] | None = None,
    timeout: float,
) -> tuple[int, R] | None:
    """Run ``runner`` over ``items`` concurrently (index = priority) and return ``(index, best result)``.

    ``score=None`` ⇒ priority mode: the earliest-index success wins, and the moment the
    contiguous prefix of resolved items holds a success the rest are cancelled (so a fast
    top-priority success short-circuits everything). ``score`` ⇒ the highest-scoring success
    wins, decided once all resolve. ``accept`` short-circuits on the first success it approves.
    Returns ``None`` if nothing succeeds before ``timeout``. Pre-sort ``items`` by a per-model
    weight to get "best weighted model" with early cancellation.
    """
    tasks: dict[asyncio.Task[R], int] = {asyncio.create_task(runner(item)): i for i, item in enumerate(items)}
    results: dict[int, R] = {}
    pending = set(tasks)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break
            for task in done:
                results[tasks[task]] = task.result()
            if accept is not None:
                hit = next((i for i, r in results.items() if is_success(r) and accept(r)), None)
                if hit is not None:
                    return hit, results[hit]
            if score is None:
                ready = first_ready_success(results, is_success)
                if ready is not None:
                    return ready
        successes = [(i, r) for i, r in results.items() if is_success(r)]
        if not successes:
            return None
        if score is None:
            return min(successes, key=lambda pair: pair[0])
        return max(successes, key=lambda pair: score(pair[1]))
    finally:
        for task in pending:
            task.cancel()


class JsonGateResponse(GateResponse):
    """A ``GateResponse`` whose ``raw_text`` has been parsed into the ``json_response`` dict.

    Produced by ``.cast_json()``. ``TypedGateResponse`` extends this, so a typed result is also a
    JSON result (raw_text + json_response + value).
    """

    json_response: JsonDict | None = None

    @classmethod
    def of(cls, response: GateResponse, json_response: JsonDict | None = None) -> JsonGateResponse:
        """Wrap a ``GateResponse`` as a ``JsonGateResponse`` carrying the parsed ``json_response``."""
        return cls.model_construct(**response.__dict__, json_response=json_response)


class TypedGateResponse[T: BaseModel](JsonGateResponse):
    """A ``JsonGateResponse`` plus the parsed Pydantic ``value`` (``None`` when parsing failed).

    Access response fields directly (``typed.raw_text``, ``typed.json_response``, ``typed.status``)
    and the parsed object via ``typed.value``.
    """

    value: T | None = None

    @classmethod
    def of(cls, response: GateResponse, json_response: JsonDict | None = None, value: T | None = None) -> TypedGateResponse[T]:
        """Wrap a ``GateResponse`` with its parsed ``json_response`` and validated ``value``."""
        return cls.model_construct(**response.__dict__, json_response=json_response, value=value)


class MediaBuilder[ResponseT: GateCallRecord](BaseModel):
    """Shared callback wiring for every request builder.

    Holds the gateway handle and both callback lists, and fires them after a response:
    universal ``usage_callback`` (``RawUsage``) plus typed ``callback`` (the full response).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: GateClient
    callbacks: list[Callable[[ResponseT], Awaitable[None]]] = Field(default_factory=list)
    usage_callbacks: list[UsageCallback] = Field(default_factory=list)
    budget_check: BudgetCheck | None = None

    def callback(self, *callbacks: Callable[[ResponseT], Awaitable[None]]) -> Self:
        """Register async callbacks fired with the full response (awaited in order)."""
        self.callbacks.extend(callbacks)
        return self

    def usage_callback(self, *callbacks: UsageCallback) -> Self:
        """Register endpoint-agnostic `RawUsage` callbacks — the same counter wires into every builder."""
        self.usage_callbacks.extend(callbacks)
        return self

    def budget(self, check: BudgetCheck) -> Self:
        """Register a pre-call budget gate; a False result denies the call with ``GateBudgetError``."""
        self.budget_check = check
        return self

    async def _gate_budget(self) -> None:
        if self.budget_check is not None and not await self.budget_check():
            raise GateBudgetError("Budget exceeded; call denied before dispatch.")

    async def _fire_usage(self, usage: RawUsage) -> None:
        for cb in self.usage_callbacks:
            await cb(usage)

    async def _fire(self, response: ResponseT) -> None:
        await self._fire_usage(response.usage)
        for cb in self.callbacks:
            await cb(response)


class RequestBuilder[ResponseT: GateResponse](MediaBuilder[GateResponse]):
    """Fluent helper from `GateClient.request`; not constructed directly.

    Generic over the *finalized* result type. The base returns raw ``GateResponse``;
    ``CastedRequestBuilder`` overrides ``_finalize`` to parse into ``TypedGateResponse[T]``,
    so every ``call`` / ``multicall`` / ``call_prefer`` / ``call_best`` variant is inherited
    and correctly typed.
    """

    def _finalize(self, response: GateResponse) -> ResponseT:
        """Hook turning a raw response into the builder's result type (identity for the base)."""
        return response  # type: ignore[return-value]  # ty: ignore[invalid-return-type]

    system_prompt: str | list[JsonDict]
    messages: list[Message]
    images: list[str]
    images_alternative: str | None = None
    specifics: RequestSpecifics
    timeout: int | None = None
    max_tries: int | None = None
    on_usage: OnUsage | None = None
    zone_selection: ZoneSelection | None = None
    operation: str = ""
    seed_routing: str | None = None
    cache_ttl: int | None = None
    cast_json_enabled: bool = False
    response_format: JsonDict | None = None
    tools: list[JsonDict] | None = None
    tool_choice: str | JsonDict | None = None
    parallel_tool_calls: bool | None = None
    tool_executor: ToolExecutor | None = None
    tool_stream_executor: StreamingToolExecutor | None = None
    tool_max_iters: int = MAX_TOOL_ITERS
    tool_concurrent: bool = True
    tool_max_tokens_before_use: int | None = None

    def cache(self, ttl: int | None) -> Self:
        """Set this call's cache TTL in seconds, overriding the client default.

        ``None`` inherits the default, ``0`` forces off, positive caches a SUCCESS that long.
        Only non-streaming paths are cached server-side.
        """
        self.cache_ttl = ttl
        return self

    def zone(self, selection: ZoneSelection | str) -> Self:
        """Restrict this call to a zone, overriding the client default.

        A ``str`` is shorthand for ``ZoneSelection.zone(value)`` (a ``provider_region`` filter);
        pass ``ZoneSelection()`` to widen back to all deployments.
        """
        self.zone_selection = ZoneSelection.zone(selection) if isinstance(selection, str) else selection
        return self

    def with_tools(
        self,
        tools: list[JsonDict],
        executor: ToolExecutor | None = None,
        *,
        stream_executor: StreamingToolExecutor | None = None,
        max_iters: int = MAX_TOOL_ITERS,
        concurrent: bool = True,
        tool_choice: str | JsonDict | None = None,
        parallel_tool_calls: bool | None = None,
        max_tokens_before_tool_use: int | None = None,
    ) -> Self:
        """Make ``tools`` available to the model on ``call`` / ``call_stream``.

        ``tools`` are OpenAI-shaped function schemas (sent to the model). With no ``executor``,
        a single call runs and the model's requested calls land in ``response.tool_calls`` for
        you to handle. With an ``executor`` ``(id, name, parsed_args) -> result_text``, the loop
        runs each call, feeds results back, and re-invokes until a turn has no tool calls or
        ``max_iters``. ``concurrent`` runs a turn's calls together (the streaming path always does).

        ``stream_executor`` (``call_stream`` only) is a streaming variant
        ``(id, name, parsed_args) -> AsyncIterator[ToolStreamItem]``: yield ``ToolProgress`` to
        surface live tool progress into the output stream and one ``ToolResult`` for the result
        fed back to the model (``ToolResult(redo=False)`` ends the loop). Takes precedence over
        ``executor`` when streaming.

        ``tool_choice`` ('auto' | 'none' | 'required' | a named-function dict) and
        ``parallel_tool_calls`` are forwarded to the model (translated for Anthropic).

        ``max_tokens_before_tool_use``: once the running conversation exceeds this many tokens,
        the loop stops offering tools and instructs the model to answer directly (prevents a
        too-large context from triggering yet another tool round). ``None`` disables the guard.
        """
        self.tools = tools
        self.tool_executor = executor
        self.tool_stream_executor = stream_executor
        self.tool_max_iters = max_iters
        self.tool_concurrent = concurrent
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_max_tokens_before_use = max_tokens_before_tool_use
        return self

    def cast(self, model_type: type[T]) -> CastedRequestBuilder[T]:
        """Return a copy of this builder that parses every result into ``model_type`` (forces JSON output)."""
        builder = CastedRequestBuilder[T].model_construct(**self.__dict__, model_type=model_type)
        if builder.response_format is None:
            builder.response_format = {"type": "json_object"}
        return builder

    def cast_json(self) -> JsonRequestBuilder[JsonGateResponse]:
        """Return a builder that parses every result's ``raw_text`` into ``json_response`` (a JSON dict).

        Sets ``response_format={'type':'json_object'}`` so providers that accept it force valid JSON;
        the result is a ``JsonGateResponse`` (``raw_text`` + ``json_response``). ``.cast(T)`` is the
        typed extension — it additionally validates the dict into ``T`` (``.value``).
        """
        builder = JsonRequestBuilder[JsonGateResponse].model_construct(**self.__dict__)
        builder.cast_json_enabled = True
        if builder.response_format is None:
            builder.response_format = {"type": "json_object"}
        return builder

    def _apply_cast_json(self, response: GateResponse) -> None:
        if self.cast_json_enabled and response.json_object is None and response.raw_text:
            response.json_object = extract_json_from_text(response.raw_text)

    async def _call_raw(self, model: str) -> GateResponse:
        """One non-streaming turn: budget gate → send → cast_json → usage → callbacks. No finalize."""
        await self._gate_budget()
        request = self._build_request(model, stream=False)
        response = await self.client._send(request)  # noqa: SLF001
        self._apply_cast_json(response)
        if self.on_usage is not None:
            await self.on_usage(response.usage)
        await self._fire(response)
        return response

    async def call(self, model: str) -> ResponseT:
        """Non-streaming completion for `model` (runs the tool loop when ``with_tools`` is set)."""
        raw = await self._call_tool_loop(model) if self.tool_executor is not None else await self._call_raw(model)
        return self._finalize(raw)

    async def call_stream(
        self,
        model: str,
        *,
        smooth: bool = False,
        server_side: bool = False,
        smooth_duration_ms: int = 10,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks for `model` (runs the tool loop when ``with_tools`` is set).

        ``smooth`` waits ``smooth_duration_ms`` after each text chunk — client-side, or
        ``server_side=True`` to pace on the gateway. ``on_usage`` fires after the final
        chunk (estimated input tokens on timeout/connection error, then re-raised).
        """
        if self.tool_stream_executor is not None:
            async for chunk in self._stream_tool_loop_streaming(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
            ):
                yield chunk
            return
        if self.tool_executor is not None:
            async for chunk in self._stream_tool_loop(model, smooth=smooth, server_side=server_side, smooth_duration_ms=smooth_duration_ms):
                yield chunk
            return
        await self._gate_budget()
        server_paces = smooth and server_side
        request = self._build_request(
            model,
            stream=True,
            smooth_server_side=server_paces,
            smooth_duration_ms=smooth_duration_ms if smooth else 0,
        )
        client_paces = smooth and not server_side
        final_usage: RawUsage | None = None
        try:
            async for chunk in self.client._stream(request):  # noqa: SLF001
                if chunk.input_tokens is not None or chunk.output_tokens is not None or chunk.provider is not None:
                    final_usage = RawUsage(
                        input_tokens=chunk.input_tokens or 0,
                        output_tokens=chunk.output_tokens or 0,
                        cached_input_tokens=chunk.cached_input_tokens or 0,
                        model=model,
                        provider=chunk.provider or "",
                        region=chunk.region or "",
                        duration_ms=chunk.duration_ms or 0,
                        ttft_ms=chunk.ttft_ms,
                        operation=self.operation,
                    )
                yield chunk
                if client_paces and chunk.text:
                    await asyncio.sleep(smooth_duration_ms / 1000)
        except (GateTimeoutError, GateConnectionError):
            estimated = estimate_input_tokens(self.system_prompt, self.messages, self.images)
            usage = RawUsage(model=model, input_tokens=estimated, estimated=True, operation=self.operation)
            if self.on_usage is not None:
                await self.on_usage(usage)
            await self._fire_usage(usage)
            raise
        usage = final_usage if final_usage is not None else RawUsage(model=model, estimated=True, operation=self.operation)
        if self.on_usage is not None:
            await self.on_usage(usage)
        await self._fire_usage(usage)

    def call_sync(self, model: str) -> ResponseT:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def multicall(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
    ) -> list[ResponseT]:
        """Parallel non-streaming calls to each model name."""
        await self._gate_budget()
        coros = [
            self.client._send(self._build_request(m, stream=False))  # noqa: SLF001
            for m in models
        ]
        responses = await execute_multicall(coros, models, timeout=timeout)  # ty:ignore[invalid-argument-type]
        for response in responses:
            self._apply_cast_json(response)
        if self.on_usage is not None:
            estimated_input = estimate_input_tokens(self.system_prompt, self.messages, self.images)
            for response in responses:
                usage = (
                    RawUsage(model=response.model, input_tokens=estimated_input, estimated=True, operation=self.operation)
                    if response.status == OutputStatus.TIMEOUT
                    else response.usage
                )
                await self.on_usage(usage)
        for response in responses:
            await self._fire(response)
        return [self._finalize(r) for r in responses]

    def multicall_sync(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
    ) -> list[ResponseT]:
        """Blocking wrapper for `multicall`."""
        return asyncio.run(self.multicall(models, timeout=timeout))

    async def multicall_stream(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
        specifics_by_model: dict[str, RequestSpecifics] | None = None,
    ) -> AsyncIterator[tuple[int, GateResponse]]:
        """Streaming variant of ``batch_multicall`` — yields one frame per model completion.

        Single HTTP request, image+prompt uploaded once. The server fans out across
        ``models`` and pushes an SSE frame each time a model completes (in completion
        order, not request order). ``index`` matches the position in ``models``.

        ``on_usage``, when set, fires once per yielded frame with the per-model
        ``RawUsage`` (estimated for timed-out models).
        """
        await self._gate_budget()
        request = GateRequest(
            target=ParallelTarget(models=models, specifics_by_model=specifics_by_model),
            system_prompt=self.system_prompt,
            messages=self.messages,
            images=self.images,
            images_alternative=self.images_alternative,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            response_format=self.response_format,
            specifics=self.specifics,
            stream=True,
            max_tries=self.max_tries,
            timeout=self.timeout,
            batch_timeout=timeout,
            zone_selection=self.zone_selection,
            operation=self.operation,
            seed_routing=self.seed_routing,
        )
        estimated_input: int | None = None
        async for frame in self.client._stream_batch(request):  # noqa: SLF001
            self._apply_cast_json(frame.response)
            if self.on_usage is not None:
                if frame.response.status == OutputStatus.TIMEOUT:
                    if estimated_input is None:
                        estimated_input = estimate_input_tokens(self.system_prompt, self.messages, self.images)
                    usage = RawUsage(model=frame.response.model, input_tokens=estimated_input, estimated=True, operation=self.operation)
                else:
                    usage = frame.response.usage
                await self.on_usage(usage)
            await self._fire(frame.response)
            yield frame.index, frame.response

    async def batch_multicall(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
        specifics_by_model: dict[str, RequestSpecifics] | None = None,
    ) -> list[ResponseT]:
        """Batch endpoint variant — one HTTP request, image+prompt uploaded ONCE.

        Server fans out across `models` and returns one ``GateResponse`` per model in
        the same order. Use this in preference to ``multicall`` when images/system_prompt
        are large; saves N× upload bytes.

        `specifics_by_model` lets you override the shared ``specifics`` per model name —
        useful when bundles mix providers with different ``extra_body`` requirements
        (e.g. Gemini 3 vs Gemini 2.5 thinking-config schemas).
        """
        await self._gate_budget()
        batch = GateRequest(
            target=ParallelTarget(models=models, specifics_by_model=specifics_by_model),
            system_prompt=self.system_prompt,
            messages=self.messages,
            images=self.images,
            images_alternative=self.images_alternative,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            response_format=self.response_format,
            specifics=self.specifics,
            max_tries=self.max_tries,
            timeout=self.timeout,
            batch_timeout=timeout,
            zone_selection=self.zone_selection,
            operation=self.operation,
            seed_routing=self.seed_routing,
        )
        responses = await self.client._send_batch(batch)  # noqa: SLF001
        for response in responses:
            self._apply_cast_json(response)
        if self.on_usage is not None:
            estimated_input = estimate_input_tokens(self.system_prompt, self.messages, self.images)
            for response in responses:
                usage = (
                    RawUsage(model=response.model, input_tokens=estimated_input, estimated=True, operation=self.operation)
                    if response.status == OutputStatus.TIMEOUT
                    else response.usage
                )
                await self.on_usage(usage)
        for response in responses:
            await self._fire(response)
        return [self._finalize(r) for r in responses]

    async def call_prefer(self, models: list[str]) -> ResponseT:
        """Try models in order; advance to the next on failure or non-SUCCESS status.

        Fallback triggers (try next model):
        - ``GateModelNotFoundError`` — model not configured in the gateway.
        - ``GateServerError`` — 5xx response from the gateway.
        - ``GateTimeoutError`` — HTTP-level timeout before the server replied.
        - Response with ``status != SUCCESS`` (server ran but reported failure).

        Immediate re-raise (not model-specific, no fallback):
        - ``GateAuthError``, ``GateCapabilityError``, ``GateConnectionError``.

        When an exception triggers a fallback, ``on_usage`` is called with
        estimated input tokens (``estimated=True``) for the failed attempt.
        When the server returns a non-SUCCESS response, ``on_usage`` was already
        called by ``call()`` with the server-reported usage; no duplicate call
        is made.

        Returns the last non-exception response (possibly non-SUCCESS) when all
        models are exhausted, or a ``NO_DEPLOYMENT`` response if ``models`` is
        empty or every attempt raised an exception.
        """
        _fallback_exc = (GateModelNotFoundError, GateServerError, GateTimeoutError)
        estimated_input = estimate_input_tokens(self.system_prompt, self.messages, self.images)
        last: ResponseT | None = None
        for model in models:
            try:
                resp = await self.call(model)
                if resp.status == OutputStatus.SUCCESS:
                    return resp
                # Non-SUCCESS: on_usage already fired by call(); record and try next.
                last = resp
            except _fallback_exc as exc:
                logger.warning("call_prefer: model=%s failed (%s), trying next", model, exc)
                if self.on_usage is not None:
                    await self.on_usage(RawUsage(model=model, input_tokens=estimated_input, estimated=True, operation=self.operation))
        if last is not None:
            return last
        return self._finalize(GateResponse(model=models[-1] if models else "", status=OutputStatus.NO_DEPLOYMENT))

    def call_prefer_sync(self, models: list[str]) -> ResponseT:
        """Blocking wrapper for `call_prefer`."""
        return asyncio.run(self.call_prefer(models))

    async def _safe_call(self, model: str) -> ResponseT:
        """``call`` that converts model-specific failures into a ``NO_DEPLOYMENT`` result."""
        try:
            return await self.call(model)
        except (GateModelNotFoundError, GateServerError, GateTimeoutError) as exc:
            logger.warning("call_best: model=%s failed (%s), excluded", model, exc)
            return self._finalize(GateResponse(model=model, status=OutputStatus.NO_DEPLOYMENT))

    async def _call_best_server(
        self, models: list[str], attribute: str, direction: Literal["greatest", "lowest"], timeout: float
    ) -> ResponseT:
        """Server-side best: one request fans out across `models`, returns the SUCCESS ranked best by `extra_attributes[attribute]`."""
        await self._gate_budget()
        request = GateRequest(
            target=BestTarget(models=models, attribute=attribute, direction=direction),
            system_prompt=self.system_prompt,
            messages=self.messages,
            images=self.images,
            images_alternative=self.images_alternative,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            response_format=self.response_format,
            specifics=self.specifics,
            max_tries=self.max_tries,
            timeout=self.timeout,
            batch_timeout=timeout,
            zone_selection=self.zone_selection,
            operation=self.operation,
            seed_routing=self.seed_routing,
        )
        response = await self.client._send(request)  # noqa: SLF001
        self._apply_cast_json(response)
        if self.on_usage is not None:
            await self.on_usage(response.usage)
        await self._fire(response)
        return self._finalize(response)

    async def call_best(
        self,
        models: list[str],
        *,
        key: Callable[[ResponseT], float] | None = None,
        accept: Callable[[ResponseT], bool] | None = None,
        greatest: str | None = None,
        lowest: str | None = None,
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
    ) -> ResponseT:
        """Run ``models`` concurrently and return the best SUCCESS, cancelling the losers.

        ``models`` is in priority order. With ``key=None`` the earliest-listed success wins —
        as soon as it (and any higher-priority models) resolve, the rest are cancelled; to pick
        the "best weighted model" with the same early cancellation, sort ``models`` by weight
        first. With ``key`` the highest-scoring success wins (decided once all resolve).
        ``accept`` short-circuits on the first success it approves. Every model that *completes*
        bills + fires callbacks via ``call``; only cancelled ones are spared. Returns a
        ``NO_DEPLOYMENT`` response if every model fails.

        ``greatest`` / ``lowest`` switch to SERVER-side selection: one request fans out across
        ``models`` and the gateway returns the SUCCESS whose model has the greatest/lowest
        ``extra_attributes[<name>]`` (no client-side weight fetching; ``key``/``accept`` ignored).
        """
        if greatest is not None and lowest is not None:
            msg = "Pass either `greatest` or `lowest`, not both."
            raise ValueError(msg)
        if not models:
            return self._finalize(GateResponse(model="", status=OutputStatus.NO_DEPLOYMENT))
        if greatest is not None:
            return await self._call_best_server(models, greatest, "greatest", timeout)
        if lowest is not None:
            return await self._call_best_server(models, lowest, "lowest", timeout)
        best = await select_best(
            models,
            self._safe_call,
            is_success=lambda r: r.status is OutputStatus.SUCCESS,
            score=key,
            accept=accept,
            timeout=timeout,
        )
        return best[1] if best is not None else self._finalize(GateResponse(model=models[-1], status=OutputStatus.NO_DEPLOYMENT))

    def call_best_sync(
        self,
        models: list[str],
        *,
        key: Callable[[ResponseT], float] | None = None,
        accept: Callable[[ResponseT], bool] | None = None,
        greatest: str | None = None,
        lowest: str | None = None,
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
    ) -> ResponseT:
        """Blocking wrapper for `call_best`."""
        return asyncio.run(self.call_best(models, key=key, accept=accept, greatest=greatest, lowest=lowest, timeout=timeout))

    async def _call_tool_loop(self, model: str) -> GateResponse:
        """Non-streaming tool loop; returns the raw final turn with usage aggregated across turns."""
        executor = self.tool_executor
        if executor is None:
            return await self._call_raw(model)
        messages = list(self.messages)
        total = RawUsage(model=model, operation=self.operation)
        last: GateResponse | None = None
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            turn = self.model_copy(update={"messages": turn_messages, "tool_executor": None, "tools": self.tools if allow_tools else None})
            response = await turn._call_raw(model)  # noqa: SLF001
            last = response
            total = total + response.usage
            text, tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                response.usage = total
                return response
            messages.append(Message.assistant_tool_calls(tool_calls, text))
            if self.tool_concurrent:
                results = await asyncio.gather(*[_run_tool_call(tc, executor) for tc in tool_calls])
            else:
                results = [await _run_tool_call(tc, executor) for tc in tool_calls]
            for tc, output in results:
                messages.append(Message.tool(tc.id, output, name=tc.function.name))
        if last is not None:
            last.usage = total
            return last
        return GateResponse(model=model, status=OutputStatus.NO_DEPLOYMENT)

    async def _stream_tool_loop(
        self,
        model: str,
        *,
        smooth: bool,
        server_side: bool,
        smooth_duration_ms: int,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming tool loop; yields each turn's chunks, runs tool calls between turns."""
        executor = self.tool_executor
        if executor is None:
            async for chunk in self.call_stream(model, smooth=smooth, server_side=server_side, smooth_duration_ms=smooth_duration_ms):
                yield chunk
            return
        messages = list(self.messages)
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            turn = self.model_copy(update={"messages": turn_messages, "tool_executor": None, "tools": self.tools if allow_tools else None})
            text_parts: list[str] = []
            acc: dict[int, ToolCall] = {}
            async for chunk in turn.call_stream(model, smooth=smooth, server_side=server_side, smooth_duration_ms=smooth_duration_ms):
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.tool_calls_delta:
                    _accumulate_tool_call_deltas(acc, chunk.tool_calls_delta)
                yield chunk
            if not acc:
                return
            tool_calls = [acc[i] for i in sorted(acc)]
            messages.append(Message.assistant_tool_calls(tool_calls, "".join(text_parts)))
            results = await asyncio.gather(*[_run_tool_call(tc, executor) for tc in tool_calls])
            for tc, output in results:
                messages.append(Message.tool(tc.id, output, name=tc.function.name))

    async def _stream_tool_loop_streaming(  # noqa: PLR0912
        self,
        model: str,
        *,
        smooth: bool,
        server_side: bool,
        smooth_duration_ms: int,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming tool loop whose executor yields live progress.

        Each turn streams the model's chunks, then runs the tools concurrently: ``ToolProgress``
        items are surfaced into the output stream as they arrive, and each tool's ``ToolResult``
        is fed back to the model. The loop repeats until a turn requests no tools, ``max_iters``
        is hit, or a tool returns ``ToolResult(redo=False)``.
        """
        executor = self.tool_stream_executor
        if executor is None:
            async for chunk in self.call_stream(model, smooth=smooth, server_side=server_side, smooth_duration_ms=smooth_duration_ms):
                yield chunk
            return
        messages = list(self.messages)
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            turn = self.model_copy(
                update={
                    "messages": turn_messages,
                    "tool_executor": None,
                    "tool_stream_executor": None,
                    "tools": self.tools if allow_tools else None,
                },
            )
            text_parts: list[str] = []
            acc: dict[int, ToolCall] = {}
            async for chunk in turn.call_stream(model, smooth=smooth, server_side=server_side, smooth_duration_ms=smooth_duration_ms):
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.tool_calls_delta:
                    _accumulate_tool_call_deltas(acc, chunk.tool_calls_delta)
                yield chunk
            if not acc:
                return
            tool_calls = [acc[i] for i in sorted(acc)]
            messages.append(Message.assistant_tool_calls(tool_calls, "".join(text_parts)))

            queue: asyncio.Queue[tuple[ToolCall, ToolStreamItem | None]] = asyncio.Queue()
            tasks = [asyncio.create_task(_drain_streaming_tool(executor, tc, queue)) for tc in tool_calls]
            answered: set[str] = set()
            retrigger = True
            done = 0
            while done < len(tasks):
                tc, item = await queue.get()
                if item is None:
                    done += 1
                elif isinstance(item, ToolProgress):
                    yield StreamChunk(text=item.content)
                else:
                    messages.append(Message.tool(tc.id, item.output or "", name=tc.function.name))
                    answered.add(tc.id)
                    if not item.redo:
                        retrigger = False
            await asyncio.gather(*tasks)
            # OpenAI requires a tool message per tool_call_id on the next turn; backfill any the tool left silent.
            messages.extend(Message.tool(tc.id, "", name=tc.function.name) for tc in tool_calls if tc.id not in answered)
            if not retrigger:
                return

    def _build_request(self, model: str, stream: bool, *, smooth_server_side: bool = False, smooth_duration_ms: int = 0) -> GateRequest:
        return GateRequest(
            target=SingleTarget(model=model),
            system_prompt=self.system_prompt,
            messages=self.messages,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            response_format=self.response_format,
            images=self.images,
            images_alternative=self.images_alternative,
            specifics=self.specifics,
            stream=stream,
            timeout=self.timeout,
            max_tries=self.max_tries,
            zone_selection=self.zone_selection,
            operation=self.operation,
            seed_routing=self.seed_routing,
            cache_ttl=self.cache_ttl,
            smooth=smooth_server_side,
            smooth_duration_ms=smooth_duration_ms,
        )


class JsonRequestBuilder[ResponseT: JsonGateResponse](RequestBuilder[ResponseT]):
    """Intermediary tier: parses each result's ``raw_text`` into ``json_response``.

    ``.cast_json()`` yields this; ``.cast(T)`` (``CastedRequestBuilder``) extends it with a typed value.
    """

    def _parsed_json(self, response: GateResponse) -> JsonDict | None:
        """The effective JSON dict: server-set ``json_object`` if present, else parsed from ``raw_text``."""
        if response.json_object is not None:
            return response.json_object
        return extract_json_from_text(response.raw_text) if response.raw_text else None

    def _finalize(self, response: GateResponse) -> ResponseT:
        return JsonGateResponse.of(response, self._parsed_json(response))  # type: ignore[return-value]  # ty: ignore[invalid-return-type]


class CastedRequestBuilder[T: BaseModel](JsonRequestBuilder[TypedGateResponse[T]]):
    """Typed builder: a JSON request that also validates the parsed dict into ``T`` (``.value``).

    Inherits every ``call`` / ``multicall`` / ``call_prefer`` / ``call_best`` variant and only
    overrides ``_finalize``. ``key`` / ``accept`` callbacks receive the typed result, so you can
    rank by ``.value``.
    """

    model_type: type[T]

    def _finalize(self, response: GateResponse) -> TypedGateResponse[T]:
        parsed = self._parsed_json(response)
        value: T | None = None
        if response.status == OutputStatus.SUCCESS and parsed is not None:
            try:
                value = self.model_type.model_validate(parsed)
            except ValidationError:
                value = None
        return TypedGateResponse.of(response, parsed, value)


# ---------------------------------------------------------------------------
# Image / Audio / Video builders
# ---------------------------------------------------------------------------


class ImageRequestBuilder(MediaBuilder[ImageResponse]):
    """Fluent helper from ``GateClient.image_request``; ``.call(model)`` / ``.call_stream(model)`` send it."""

    request: ImageRequest

    async def call(self, model: str) -> ImageResponse:
        """Generate or edit images with ``model`` (edit mode when ``images`` was set)."""
        await self._gate_budget()
        response = await self.client._send_images(self.request.model_copy(update={"model": model}))  # noqa: SLF001
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> ImageResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def call_stream(self, model: str) -> AsyncIterator[ImageData]:
        """Stream partial image frames; after the final frame, fire usage + result callbacks."""
        await self._gate_budget()
        request = self.request.model_copy(update={"model": model})
        frames: list[ImageData] = []
        terminal_usage: RawUsage | None = None
        async for chunk in self.client._stream_images(request):  # noqa: SLF001
            if isinstance(chunk, ImageData):
                frames.append(chunk)
                yield chunk
            else:
                terminal_usage = chunk
        synth = ImageResponse(model=request.model, data=frames, usage=terminal_usage or RawUsage(model=request.model))
        await self._fire(synth)


class TTSRequestBuilder(MediaBuilder[BaseAudioResponse]):
    """Speech builder from ``GateClient.audio_request(mode='speech')``; supports ``.call`` and ``.call_stream``."""

    request: TTSRequest

    async def call(self, model: str) -> BaseAudioResponse:
        """Synthesize speech with ``model`` (base64-encoded audio + usage)."""
        await self._gate_budget()
        response = await self.client._send_tts(self.request.model_copy(update={"model": model}))  # noqa: SLF001
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> BaseAudioResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def call_stream(self, model: str) -> AsyncIterator[bytes]:
        """Stream raw speech bytes; fire usage (``input_tokens == len(text)``) at completion."""
        await self._gate_budget()
        request = self.request.model_copy(update={"model": model})
        async for chunk in self.client._stream_tts(request):  # noqa: SLF001
            yield chunk
        synth = TTSResponse(
            model=request.model,
            audio="",
            response_format=request.response_format,
            usage=RawUsage(input_tokens=len(request.text), model=request.model),
        )
        await self._fire(synth)


class AudioGenRequestBuilder(MediaBuilder[BaseAudioResponse]):
    """Generative-audio builder from ``GateClient.audio_request(mode='music'|'sound_effects'|'dialogue')``."""

    request: AudioGenRequest

    async def call(self, model: str) -> BaseAudioResponse:
        """Generate music / sound effects / dialogue with ``model`` (base64-encoded audio + usage)."""
        await self._gate_budget()
        response = await self.client._send_audio_gen(self.request.model_copy(update={"model": model}))  # noqa: SLF001
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> BaseAudioResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))


class VideoRequestBuilder(MediaBuilder[VideoResponse]):
    """Fluent helper from ``GateClient.video_request``; ``.call(model)`` sends it (long-running, no stream)."""

    request: VideoRequest

    async def call(self, model: str) -> VideoResponse:
        """Generate a video with ``model`` and parse the response (base64-encoded video + usage)."""
        await self._gate_budget()
        request = self.request.model_copy(update={"model": model})
        response = await self.client._send_video(request)  # noqa: SLF001
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> VideoResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))


class DirectRequestBuilder[ResponseT: GateCallRecord](MediaBuilder[ResponseT]):
    """One-shot builder for the non-streaming data endpoints (embed / transcribe / isolation / dub / vision / responses).

    Carries only its endpoint config; ``call`` resolves the model then defers budget gating,
    usage, and callbacks to the shared ``MediaBuilder`` machinery — same as every other builder.
    """

    request: BaseModel
    path: str
    label: str
    response_type: type[ResponseT]
    client_timeout: float | None = None

    async def call(self, model: str) -> ResponseT:
        """POST to the endpoint with ``model`` resolved, then fire usage + callbacks."""
        await self._gate_budget()
        request = self.request.model_copy(update={"model": model})
        response = await self.client._post_json(self.path, request, self.label, client_timeout=self.client_timeout)  # noqa: SLF001
        parsed = self.response_type.model_validate(response.json())
        await self._fire(parsed)
        return parsed

    def call_sync(self, model: str) -> ResponseT:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))
