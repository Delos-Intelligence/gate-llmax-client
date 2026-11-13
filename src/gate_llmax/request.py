"""RequestBuilder, CastedRequestBuilder, and TypedLLMResponse for the Gate client."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast, get_origin, overload
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from gate_llmax.ids import new_call_id
from gate_llmax.models.audio_gen import AudioGenRequest
from gate_llmax.models.images import ImageData, ImageRequest, ImageResponse
from gate_llmax.models.messages import ImageMessage, Message, TextMessage
from gate_llmax.models.request import (
    BestTarget,
    CompactionPolicy,
    ContextEdits,
    FallbackTarget,
    LLMRequest,
    OnDemandCompaction,
    ParallelTarget,
    RequestSpecifics,
    SingleTarget,
    ZoneSelection,
)
from gate_llmax.models.response import BaseAudioResponse, LLMCallRecord, LLMResponse, RawUsage, StreamChunk, ToolCall
from gate_llmax.models.tts import TTSRequest, TTSResponse
from gate_llmax.models.video import VideoRequest, VideoResponse
from gate_llmax.types import JsonDict, OutputStatus

from .exceptions import (
    LLMBudgetError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMServerError,
    LLMTimeoutError,
)
from .metering import (
    LOST_AFTER_SEND,
    Closing,
    MeteredStream,
    StreamMeter,
    bears_usage,
    chunk_usage,
    estimated_usage,
    left_settling,
    settle,
    timed_out_unsent,
)
from .multicall import execute_multicall
from .parsing import extract_json_from_text, extract_json_list_from_text
from .tokens import count, estimate_input_tokens

if TYPE_CHECKING:
    from .client import LLMClient

logger = logging.getLogger("gate.client.request")

T = TypeVar("T", bound=BaseModel)
# Unbound: list elements go through TypeAdapter, which takes primitives and dicts too.
ElemT = TypeVar("ElemT")

_DEFAULT_MULTICALL_TIMEOUT = 60.0

OnUsage = Callable[[RawUsage], Awaitable[None]]
UsageCallback = Callable[[RawUsage], Awaitable[None]]
"""Endpoint-agnostic callback. Fires once per response with that response's `RawUsage`."""

LLMCallback = Callable[[LLMResponse], Awaitable[None]]
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

    ``output`` becomes the tool message content, ``images`` ride along in the same message
    for the model to see. ``redo`` controls whether the model is re-invoked after the tools
    run — ``False`` ends the loop (e.g. the tool already produced the final answer and there
    is nothing left for the model to do). ``add_tools`` are function schemas offered to the
    model from the next turn on, so a tool can load others on demand.
    """

    output: str | None = None
    images: list[ImageMessage] = Field(default_factory=list)
    redo: bool = True
    add_tools: list[dict[str, Any]] = Field(default_factory=list)


ToolStreamItem = ToolProgress | ToolResult
StreamingToolExecutor = Callable[[str, str, dict[str, Any]], AsyncIterator[ToolStreamItem]]
"""Streams one tool call: ``(tool_call_id, name, parsed_args) -> AsyncIterator[ToolStreamItem]``.

Yield ``ToolProgress`` items to surface live progress into the output stream, and one
``ToolResult`` to feed the result back to the model (``redo=False`` ends the loop)."""

BudgetCheck = Callable[[], Awaitable[bool]]
"""Pre-call budget gate: returns True to allow the call, False to deny it with `LLMBudgetError`."""

MAX_TOOL_ITERS = 8

_TOOL_BUDGET_EXCEEDED_NOTE = (
    "You can no longer call any tools due to the size of the context. With the current information, answer the user's question directly."
)


def _conversation_token_count(messages: list[Message]) -> int:
    """Exact token count of all text content across ``messages`` (for the tool-budget guard)."""
    return sum(count(block.text) for msg in messages for block in msg.content if isinstance(block, TextMessage))


def _extract_tool_calls(response: LLMResponse) -> tuple[str, list[ToolCall] | None]:
    """Pull (assistant_text, tool_calls) out of a ``LLMResponse``."""
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
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        async for item in executor(tool_call.id, tool_call.function.name, args):
            await queue.put((tool_call, item))
    finally:
        # Also when the executor raises: the loop waits for every ``None``, then its gather re-raises.
        queue.put_nowait((tool_call, None))


def _tool_schema_name(tool: Mapping[str, Any]) -> str | None:
    function = tool.get("function")
    return function.get("name") if isinstance(function, Mapping) else None


def _unoffered_tools(offered: list[dict[str, Any]], added: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {_tool_schema_name(tool) for tool in offered}
    fresh: list[dict[str, Any]] = []
    for tool in added:
        if (name := _tool_schema_name(tool)) is not None and name not in names:
            names.add(name)
            fresh.append(dict(tool))
    return fresh


@contextmanager
def _naming_failures(call_id: UUID) -> Iterator[None]:
    """Stamp a dispatch's call id on the ``LLMError`` it raises, so whoever catches it knows which Gate request failed."""
    try:
        yield
    except LLMError as exc:
        if exc.call_id is None:
            exc.call_id = call_id
        raise


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
    Returns ``None`` if nothing succeeds before ``timeout``. Either way it returns only once the
    runners it cancelled have finished, their usage fired. Pre-sort ``items`` by a per-model
    weight to get "best weighted model" with early cancellation.
    """
    with left_settling() as settling:
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
        # A cancelled runner still bills on its way out (its estimate, or Gate's usage it was handing out):
        # wait for it, or asyncio.run cuts it off.
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*settling, return_exceptions=True)


def _parse_json(text: str, *, loose: bool, repair: bool = False) -> JsonDict | None:
    """Parse ``text`` to a JSON dict. ``loose`` brace-slices around surrounding prose / fences; else strict.

    ``repair`` adds an aggressive control-char / bad-escape repair pass for malformed JSON.
    """
    if loose or repair:
        return extract_json_from_text(text, repair=repair)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


class JsonLLMResponse(LLMResponse):
    """A ``LLMResponse`` whose ``raw_text`` has been parsed into the ``json_response`` dict.

    Produced by ``.cast_json()``. ``TypedLLMResponse`` extends this, so a typed result is also a
    JSON result (raw_text + json_response + value).
    """

    # ``dict[str, Any]`` (not ``JsonDict``) so call sites use values without casting, like ``json.loads``.
    json_response: dict[str, Any] | None = None

    @classmethod
    def of(cls, response: LLMResponse, json_response: dict[str, Any] | None = None) -> JsonLLMResponse:
        """Wrap a ``LLMResponse`` as a ``JsonLLMResponse`` carrying the parsed ``json_response``."""
        return cls.model_construct(**response.__dict__, json_response=json_response)


class TypedLLMResponse[T](JsonLLMResponse):
    """A ``JsonLLMResponse`` plus the parsed Pydantic ``value`` (``None`` when parsing failed).

    ``T`` is a Pydantic model (``.cast(Model)``) or a list of them (``.cast(list[Model])``),
    so ``value`` is the model or ``list`` of models.

    Access response fields directly (``typed.raw_text``, ``typed.json_response``, ``typed.status``)
    and the parsed object via ``typed.value``.
    """

    value: T | None = None

    @classmethod
    def of(cls, response: LLMResponse, json_response: dict[str, Any] | None = None, value: T | None = None) -> TypedLLMResponse[T]:
        """Wrap a ``LLMResponse`` with its parsed ``json_response`` and validated ``value``."""
        return cls.model_construct(**response.__dict__, json_response=json_response, value=value)


class MediaBuilder[ResponseT: LLMCallRecord](BaseModel):
    """Shared callback wiring for every request builder.

    Holds the gateway handle and both callback lists, and fires them after a response:
    universal ``usage_callback`` (``RawUsage``) plus typed ``callback`` (the full response).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: LLMClient
    callbacks: list[Callable[[ResponseT], Awaitable[None]]] = Field(default_factory=list)
    usage_callbacks: list[UsageCallback] = Field(default_factory=list)
    budget_check: BudgetCheck | None = None
    # The caller's own id for the next dispatch (``with_call_id``); taken by it, so the ones after mint their own.
    call_id: UUID | None = None

    def callback(self, *callbacks: Callable[[ResponseT], Awaitable[None]]) -> Self:
        """Register async callbacks fired with the full response (awaited in order)."""
        self.callbacks.extend(callbacks)
        return self

    def usage_callback(self, *callbacks: UsageCallback) -> Self:
        """Register endpoint-agnostic `RawUsage` callbacks — the same counter wires into every builder."""
        self.usage_callbacks.extend(callbacks)
        return self

    def budget(self, check: BudgetCheck) -> Self:
        """Register a pre-call budget gate; a False result denies the call with ``LLMBudgetError``."""
        self.budget_check = check
        return self

    def with_call_id(self, call_id: UUID) -> Self:
        """Send the next dispatch to Gate as ``call_id`` instead of a fresh UUIDv7; the dispatches after it mint their own.

        A dispatch is one Gate request: a tool loop's turns, ``multicall``'s models and ``call_prefer``'s
        attempts are each their own, so only the first of them takes this id.
        """
        self.call_id = call_id
        return self

    def _take_call_id(self) -> UUID | None:
        """Hand the caller's ``with_call_id`` id over once — to a dispatch, or to the builder copy about to make it."""
        call_id, self.call_id = self.call_id, None
        return call_id

    def _next_call_id(self) -> UUID:
        """The call id of the dispatch about to go out: the caller's own, once, else a fresh UUIDv7 minted now."""
        return self._take_call_id() or new_call_id()

    async def _gate_budget(self) -> None:
        if self.budget_check is not None and not await self.budget_check():
            raise LLMBudgetError("Budget exceeded; call denied before dispatch.")

    async def _fire_usage(self, usage: RawUsage) -> None:
        self.client._record_usage(usage)  # noqa: SLF001  (per-run cost accumulator → client.total_usage)
        for cb in self.usage_callbacks:
            await cb(usage)

    async def _fire(self, response: ResponseT, *, disable_usage: bool = False) -> None:
        if not disable_usage:
            await self._fire_usage(response.usage)
        for cb in self.callbacks:
            await cb(response)

    async def call(self, model: str) -> ResponseT:
        """Send the request for ``model``. Each builder overrides this."""
        raise NotImplementedError

    async def call_prefer(self, models: list[str]) -> ResponseT:
        """Try ``models`` in order, returning the first success — sequential, so only the winner bills.

        Unlike a fan-out (which would call every model and bill them all), this runs one model at a
        time and stops at the first success. Set per-attempt limits via the request's ``timeout`` /
        ``max_tries``. The last error is re-raised when every model fails.
        """
        last_error: Exception | None = None
        for model in models:
            try:
                return await self.call(model)
            except (LLMModelNotFoundError, LLMServerError, LLMTimeoutError, LLMConnectionError, TimeoutError) as exc:
                logger.warning("call_prefer: model=%s failed (%s), trying next", model, exc)
                last_error = exc
        if last_error is not None:
            raise last_error
        msg = "call_prefer requires at least one model."
        raise ValueError(msg)


class RequestBuilder[ResponseT: LLMResponse](MediaBuilder[LLMResponse]):
    """Fluent helper from `LLMClient.request`; not constructed directly.

    Generic over the *finalized* result type. The base returns raw ``LLMResponse``;
    ``CastedRequestBuilder`` overrides ``_finalize`` to parse into ``TypedLLMResponse[T]``,
    so every ``call`` / ``multicall`` / ``call_prefer`` / ``call_best`` variant is inherited
    and correctly typed.
    """

    def _finalize(self, response: LLMResponse) -> ResponseT:
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
    hosting_providers: list[str] | None = None
    plan: str | None = None
    deployment_id: str | None = None
    no_fallback: bool = False
    fallback_on_content_policy: bool = False
    context_edits: ContextEdits | None = None
    compaction: CompactionPolicy | None = None
    compact_now: OnDemandCompaction | None = None
    operation: str = ""
    seed_routing: str | None = None
    # When set, every streamed call this builder makes (including each tool-loop turn) sends a
    # FallbackTarget(prefer_models) instead of a SingleTarget: the gateway tries the models in
    # order per inference — one request, server-side — preferring prefer_models[0]. Set via
    # RequestBuilder.stream(...).call_prefer(...); propagated by model_copy.
    prefer_models: list[str] | None = None
    cache_call: bool | None = None
    cache_ttl: int | None = None
    cast_json_enabled: bool = False
    loose_edges: bool = True
    repair: bool = False
    response_format: JsonDict | None = None
    tools: list[JsonDict] | None = None
    tool_choice: str | JsonDict | None = None
    parallel_tool_calls: bool | None = None
    tool_executor: ToolExecutor | None = None
    tool_stream_executor: StreamingToolExecutor | None = None
    tool_max_iters: int = MAX_TOOL_ITERS
    tool_concurrent: bool = True
    tool_max_tokens_before_use: int | None = None

    def cache(self, ttl: int | None = None, *, call: bool | None = True) -> Self:
        """Turn the gateway response cache on for this call, overriding the client default.

        ``call`` decides whether to cache: ``True`` (the default) forces it on, ``False`` off,
        ``None`` defers to the API key's ``response_caching``. ``ttl`` only sets the lifetime of a
        cached SUCCESS — ``None`` uses the gateway default. Streaming calls cache too, replaying
        their recorded chunks.
        """
        if ttl is not None and ttl <= 0:
            # cache(0) used to force caching off; under cache_call it would silently mean
            # "on, at the gateway default lifetime". Reject it rather than invert its meaning.
            raise ValueError("cache(ttl=) must be positive; pass cache(call=False) to force caching off.")
        self.cache_call = call
        self.cache_ttl = ttl
        return self

    def zone(self, selection: ZoneSelection | str) -> Self:
        """Restrict this call to a zone, overriding the client default.

        A ``str`` is shorthand for ``ZoneSelection.zone(value)`` (a ``provider_region`` filter);
        pass ``ZoneSelection()`` to widen back to all deployments.
        """
        self.zone_selection = ZoneSelection.zone(selection) if isinstance(selection, str) else selection
        return self

    def hosting(self, *providers: str) -> Self:
        """Restrict this call to deployments on these hosting providers, overriding the client default.

        A canonical slug also admits its tier variants (``"azure"`` includes ``"azure-cheap"``);
        a variant admits only itself (``"azure-cheap"`` never widens to plain ``"azure"``).
        Calling with no arguments widens back to all hosting providers. Composes with
        ``.zone(...)`` (AND).
        """
        self.hosting_providers = list(providers) if providers else None
        return self

    def dev(self, deployment_id: str) -> Self:
        """Dev keys only: serve from this deployment id alone, skipping filters, rotation and fallback."""
        self.deployment_id = deployment_id
        return self

    def without_fallback(self, *, on: bool = True) -> Self:
        """Refuse the automatic model-level fallback chain; fail rather than serve a different model."""
        self.no_fallback = on
        return self

    def with_content_policy_fallback(self, *, on: bool = True) -> Self:
        """On a provider content-policy block, advance to the next fallback model rather than return the block."""
        self.fallback_on_content_policy = on
        return self

    def with_context_edits(self, edits: ContextEdits) -> Self:
        """Clear old tool results and reasoning once the prompt grows past the edit's trigger."""
        self.context_edits = edits
        return self

    def with_compaction(self, policy: CompactionPolicy) -> Self:
        """Summarize the history once the prompt passes the policy's trigger; the reply then carries the ``Compaction``."""
        self.compaction = policy
        return self

    def compact(self, instructions: str | None = None, *, native: bool = True) -> Self:
        """Ask for a summary of the history instead of a reply; open the next history with ``Message.compacted(response.compaction)``."""
        self.compact_now = OnDemandCompaction(instructions=instructions, native=native)
        return self

    def _conversation_controls(self) -> dict[str, Any]:
        """The context edits and compaction every request this builder sends carries."""
        return {"context_edits": self.context_edits, "compaction": self.compaction, "compact": self.compact_now}

    def with_tools(
        self,
        tools: Sequence[Mapping[str, Any]],
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
        self.tools = [dict(t) for t in tools]
        self.tool_executor = executor
        self.tool_stream_executor = stream_executor
        self.tool_max_iters = max_iters
        self.tool_concurrent = concurrent
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_max_tokens_before_use = max_tokens_before_tool_use
        return self

    @overload
    def cast(self, model_type: type[T]) -> CastedRequestBuilder[T]: ...
    @overload
    def cast(self, model_type: type[list[ElemT]]) -> CastedRequestBuilder[list[ElemT]]: ...
    def cast(self, model_type: Any) -> CastedRequestBuilder[Any]:
        """Return a copy of this builder that parses every result into ``model_type`` (forces JSON output).

        ``model_type`` is a Pydantic model, or ``list[Model]`` to parse a top-level JSON
        array into a ``list`` of that model (each element validated). ``.value`` is ``None``
        when nothing parses.
        """
        builder = CastedRequestBuilder[Any].model_construct(**self.__dict__, model_type=model_type)
        builder.call_id = self._take_call_id()  # the copy is what dispatches now, so the caller's id moves to it
        if builder.response_format is None:
            builder.response_format = {"type": "json_object"}
        return builder

    def cast_json(self, *, loose_edges: bool = True, repair: bool = False) -> JsonRequestBuilder[JsonLLMResponse]:
        """Return a builder that parses every result's ``raw_text`` into ``json_response`` (a JSON dict).

        Sets ``response_format={'type':'json_object'}`` so providers that accept it force valid JSON;
        the result is a ``JsonLLMResponse`` (``raw_text`` + ``json_response``, ``None`` when nothing
        parsed — the response object is preserved, never raised). ``.cast(T)`` is the typed extension.

        ``loose_edges`` (default ``True``) tolerates prose / markdown fences around the JSON
        (brace-slices the first ``{...}``); set ``False`` to require the whole reply to be valid JSON.
        ``repair`` (default ``False``) adds an aggressive control-char / bad-escape repair pass for
        models that emit slightly malformed JSON.
        """
        builder = JsonRequestBuilder[JsonLLMResponse].model_construct(**self.__dict__)
        builder.call_id = self._take_call_id()  # the copy is what dispatches now, so the caller's id moves to it
        builder.cast_json_enabled = True
        builder.loose_edges = loose_edges
        builder.repair = repair
        if builder.response_format is None:
            builder.response_format = {"type": "json_object"}
        return builder

    def _apply_cast_json(self, response: LLMResponse) -> None:
        if self.cast_json_enabled and response.json_object is None and response.raw_text:
            response.json_object = _parse_json(response.raw_text, loose=self.loose_edges, repair=self.repair)

    async def _bill(self, usage: RawUsage) -> None:
        """Hand one dispatch's usage to ``on_usage``, then to the usage callbacks and the client's total."""
        if self.on_usage is not None:
            await self.on_usage(usage)
        await self._fire_usage(usage)

    async def _estimate(
        self,
        model: str,
        call_id: UUID | None,
        input_tokens: int,
        output_tokens: int = 0,
        *,
        served: str | None = None,
        finish_reason: str = "",
    ) -> RawUsage:
        """A usage Gate never reported, at the list price of the model that served (``served``), else of ``model``.

        A group is priced as the model it serves on this request's plan. The cost is 0 when Gate
        can't say in time; never raises.
        """
        price = await self.client._list_price(served or model, self.plan)  # noqa: SLF001
        return estimated_usage(
            price,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=self.operation,
            call_id=call_id,
            finish_reason=finish_reason,
        )

    async def _bill_estimates(self, models: Sequence[str], call_id: UUID) -> None:
        """Bill a dispatch Gate never answered: an estimate of its prompt, no output, for each of ``models`` it ran."""
        prompt = estimate_input_tokens(self.system_prompt, self.messages, self.images)
        for usage in await asyncio.gather(*(self._estimate(model, call_id, prompt) for model in models)):
            await self._bill(usage)

    async def _bill_answers(self, responses: Sequence[LLMResponse]) -> None:
        """Bill the answers Gate returned: its usage, a timed-out answer's as a priced estimate of its prompt."""
        await self._estimate_timeouts(responses)
        for response in responses:
            await self._bill(response.usage)

    async def _bill_stream(self, meter: StreamMeter, model: str, call_id: UUID) -> None:
        """Bill a streamed dispatch on its way out: the legs Gate reported, else an estimate of every leg."""
        if meter.reported:
            usage = meter.reported_usage(model=model, operation=self.operation, call_id=call_id)
        else:
            prompt = estimate_input_tokens(self.system_prompt, self.messages, self.images)
            input_tokens, output_tokens = meter.estimated_tokens(prompt)
            usage = await self._estimate(
                model, call_id, input_tokens, output_tokens, served=meter.served_model, finish_reason=meter.finish_reason
            )
        await self._bill(usage)

    async def _estimate_timeouts(self, responses: Sequence[LLMResponse]) -> None:
        """Swap each timed-out answer's usage for a priced estimate of its prompt, under the call id it carries."""
        timed_out = [response for response in responses if response.status == OutputStatus.TIMEOUT]
        if not timed_out:
            return
        prompt = estimate_input_tokens(self.system_prompt, self.messages, self.images)
        estimates = await asyncio.gather(*(self._estimate(r.model, r.usage.call_id, prompt) for r in timed_out))
        for response, usage in zip(timed_out, estimates, strict=True):
            response.usage = usage

    async def _call_raw(self, model: str, *, priority: int = 0, disable_usage: bool = False) -> LLMResponse:
        """One non-streaming turn: budget gate → send → cast_json → usage → callbacks. No finalize.

        ``disable_usage`` skips the usage callbacks (``on_usage`` + billing) for this turn;
        response callbacks still fire. A turn that times out, is cancelled or loses its connection
        before Gate answers fires an estimate of its prompt (``estimated=True``), then re-raises;
        one that could not connect, or timed out before it was sent, fires nothing, Gate never
        having received it.
        """
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self._build_request(model, stream=False, call_id=call_id)
        try:
            with _naming_failures(call_id):
                response = await self.client._send(request, priority=priority)  # noqa: SLF001
        except (LLMTimeoutError, asyncio.CancelledError, *LOST_AFTER_SEND) as exc:
            # No answer, so no usage from Gate; a request that went out bills its prompt as an estimate.
            if not disable_usage and not timed_out_unsent(exc):
                await settle(self._bill_estimates([model], call_id), quiet=True)
            raise
        response.usage.call_id = call_id
        self._apply_cast_json(response)
        if not disable_usage:
            # Gate answered and billed: its usage lands even if the caller is cancelled while it is handed out.
            await settle(self._bill(response.usage), quiet=False)
        await self._fire(response, disable_usage=True)
        return response

    async def call(self, model: str, *, priority: int = 0, disable_usage: bool = False) -> ResponseT:
        """Non-streaming completion for `model` (runs the tool loop when ``with_tools`` is set).

        ``priority`` orders calls that queue for a rate-limit concurrency slot (higher first).
        ``disable_usage`` skips the usage callbacks for this call only, so a billing-enabled
        client can make an unbilled call without ``clear_usage_callbacks()``.
        """
        raw = (
            await self._call_tool_loop(model, disable_usage=disable_usage)
            if self.tool_executor is not None
            else await self._call_raw(model, priority=priority, disable_usage=disable_usage)
        )
        return self._finalize(raw)

    async def call_stream(
        self,
        model: str,
        *,
        smooth: bool = False,
        server_side: bool = False,
        smooth_duration_ms: int = 10,
        priority: int = 0,
        disable_usage: bool = False,
        usage_chunks: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks for `model` (runs the tool loop when ``with_tools`` is set).

        ``smooth`` waits ``smooth_duration_ms`` after each text chunk — client-side, or
        ``server_side=True`` to pace on the gateway.
        ``disable_usage`` skips the usage callbacks for this stream only, so a billing-enabled
        client can make an unbilled call without ``clear_usage_callbacks()``.
        ``usage_chunks`` controls when the usage callbacks fire: ``False`` (default) fires once at
        the end with the final usage (one DB write per stream); ``True`` fires on every usage-bearing
        chunk. Either way the chunks always carry their usage fields.

        The usage fires on every exit of a dispatch Gate received: the end, the caller stopping
        (``aclose()``), a cancellation, a timeout, a cut-off, or an error after chunks arrived. It
        is Gate's report, summed over the legs of a resumed stream that Gate reported (a leg dropped
        before its report is not counted); when Gate reported no leg, an estimate
        (``estimated=True``): the prompt each leg was sent and the output it delivered, at the
        model's list price. A caller that ``break``s out of ``async for`` without closing
        the stream leaves it to the event loop, so its usage fires when the loop finalizes it.
        """
        if self.tool_stream_executor is not None:
            looped = self._stream_tool_loop_streaming(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            )
            async with Closing(looped) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        if self.tool_executor is not None:
            looped = self._stream_tool_loop(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            )
            async with Closing(looped) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        await self._gate_budget()
        server_paces = smooth and server_side
        call_id = self._next_call_id()
        request = self._build_request(
            model,
            stream=True,
            call_id=call_id,
            smooth_server_side=server_paces,
            smooth_duration_ms=smooth_duration_ms if smooth else 0,
        )
        client_paces = smooth and not server_side
        stream = MeteredStream.of(self.client._stream(request, priority=priority))  # noqa: SLF001
        error: BaseException | None = None
        try:
            with _naming_failures(call_id):
                async with Closing(stream) as chunks:
                    async for chunk in chunks:
                        if usage_chunks and not disable_usage and bears_usage(chunk):
                            usage = chunk_usage(chunk, model=model, operation=self.operation, call_id=call_id)
                            await settle(self._bill(usage), quiet=False)
                        yield chunk
                        if client_paces and chunk.text:
                            await asyncio.sleep(smooth_duration_ms / 1000)
        except BaseException as exc:
            error = exc
            raise
        finally:
            # One usage per dispatch, however it ends; ``usage_chunks`` already fired Gate's, so it only adds a missing estimate.
            meter = stream.meter
            if not disable_usage and meter.owes_usage(error) and not (usage_chunks and meter.reported):
                await settle(self._bill_stream(meter, model, call_id), quiet=error is not None)

    def stream(
        self,
        *,
        smooth: bool = False,
        server_side: bool = False,
        smooth_duration_ms: int = 10,
        priority: int = 0,
        disable_usage: bool = False,
        usage_chunks: bool = False,
    ) -> StreamRequestBuilder:
        """Enter streaming mode — the streamed counterpart of this builder.

        Streaming / pacing options are captured here; model selection stays on the returned
        builder's terminals (``.call(model)`` / ``.call_prefer(models)``), which yield
        ``StreamChunk``s. ``call_prefer`` reuses the same server-side ``FallbackTarget``
        mechanism as a non-streamed ``call_prefer`` — one request, no client-side re-send.
        """
        return StreamRequestBuilder(
            self,
            smooth=smooth,
            server_side=server_side,
            smooth_duration_ms=smooth_duration_ms,
            priority=priority,
            disable_usage=disable_usage,
            usage_chunks=usage_chunks,
        )

    def call_sync(self, model: str) -> ResponseT:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def multicall(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
    ) -> list[ResponseT]:
        """Parallel non-streaming calls to each model name.

        A model that times out comes back ``TIMEOUT`` and bills an estimate of its prompt. One that
        fails otherwise (refused, never connected, or its connection lost) comes back ``TIMEOUT`` too
        but bills nothing; a plain ``call()`` that loses its connection bills an estimate.
        """
        await self._gate_budget()
        # One request per model, so one call id each.
        requests = [self._build_request(m, stream=False, call_id=self._next_call_id()) for m in models]
        unbilled: set[UUID | None] = set()

        async def send(request: LLMRequest) -> LLMResponse:
            try:
                return await self.client._send(request)  # noqa: SLF001
            except Exception as exc:
                if not isinstance(exc, LLMTimeoutError) or timed_out_unsent(exc):
                    unbilled.add(request.call_id)
                raise

        responses = await execute_multicall([send(request) for request in requests], models, timeout=timeout)
        for request, response in zip(requests, responses, strict=True):
            response.usage.call_id = request.call_id
            self._apply_cast_json(response)
        await settle(self._bill_answers([r for r in responses if r.usage.call_id not in unbilled]), quiet=False)
        for response in responses:
            await self._fire(response, disable_usage=True)
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
    ) -> AsyncIterator[tuple[int, LLMResponse]]:
        """Streaming variant of ``batch_multicall`` — yields one frame per model completion.

        Single HTTP request, image+prompt uploaded once. The server fans out across
        ``models`` and pushes an SSE frame each time a model completes (in completion
        order, not request order). ``index`` matches the position in ``models``.

        ``on_usage``, when set, fires once per yielded frame with the per-model
        ``RawUsage`` (a priced estimate for timed-out models, also on the frame's response).
        A stream left before every model answered (the caller stops reading, is cancelled, the
        stream times out or loses its connection, or an error follows an answer) bills each model
        still unanswered an estimate of its prompt, under the stream's call id; one Gate never
        received or refused bills nothing.
        """
        await self._gate_budget()
        # One request fanned out server-side: every model's answer shares its call id.
        call_id = self._next_call_id()
        request = LLMRequest(
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
            call_id=call_id,
            zone_selection=self.zone_selection,
            hosting_providers=self.hosting_providers,
            plan=self.plan,
            operation=self.operation,
            deployment_id=self.deployment_id,
            no_fallback=self.no_fallback,
            fallback_on_content_policy=self.fallback_on_content_policy,
            seed_routing=self.seed_routing,
            **self._conversation_controls(),
        )
        unanswered = set(range(len(models)))
        try:
            with _naming_failures(call_id):
                async with Closing(self.client._stream_batch(request)) as frames:  # noqa: SLF001
                    async for frame in frames:
                        unanswered.discard(frame.index)  # its own usage fires now, so no estimate on the way out
                        frame.response.usage.call_id = call_id
                        self._apply_cast_json(frame.response)
                        await settle(self._bill_answers([frame.response]), quiet=False)
                        await self._fire(frame.response, disable_usage=True)
                        yield frame.index, frame.response
        except BaseException as exc:
            # Left early: the models Gate was still running bill an estimate of their prompt. It ran
            # them once it answered one, or when the stream was cut after the send.
            received = len(unanswered) < len(models) or isinstance(
                exc, (GeneratorExit, asyncio.CancelledError, LLMTimeoutError, *LOST_AFTER_SEND)
            )
            if unanswered and received and not timed_out_unsent(exc):
                await settle(self._bill_estimates([models[i] for i in sorted(unanswered)], call_id), quiet=True)
            raise

    async def batch_multicall(
        self,
        models: list[str],
        timeout: float = _DEFAULT_MULTICALL_TIMEOUT,
        specifics_by_model: dict[str, RequestSpecifics] | None = None,
    ) -> list[ResponseT]:
        """Batch endpoint variant — one HTTP request, image+prompt uploaded ONCE.

        Server fans out across `models` and returns one ``LLMResponse`` per model in
        the same order. Use this in preference to ``multicall`` when images/system_prompt
        are large; saves N× upload bytes.

        `specifics_by_model` lets you override the shared ``specifics`` per model name —
        useful when bundles mix providers with different ``extra_body`` requirements
        (e.g. Gemini 3 vs Gemini 2.5 thinking-config schemas).

        A batch that times out, is cancelled or loses its connection before Gate answers bills
        each model an estimate of its prompt under the batch's call id, then re-raises; one Gate
        never received or refused bills nothing.
        """
        await self._gate_budget()
        # One request fanned out server-side: every model's answer shares its call id.
        call_id = self._next_call_id()
        batch = LLMRequest(
            target=ParallelTarget(models=models, specifics_by_model=specifics_by_model),
            call_id=call_id,
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
            hosting_providers=self.hosting_providers,
            plan=self.plan,
            operation=self.operation,
            deployment_id=self.deployment_id,
            no_fallback=self.no_fallback,
            fallback_on_content_policy=self.fallback_on_content_policy,
            seed_routing=self.seed_routing,
            **self._conversation_controls(),
        )
        try:
            with _naming_failures(call_id):
                responses = await self.client._send_batch(batch)  # noqa: SLF001
        except (LLMTimeoutError, asyncio.CancelledError, *LOST_AFTER_SEND) as exc:
            # No answer, so no usage from Gate; a batch that went out bills every model its prompt as an estimate.
            if not timed_out_unsent(exc):
                await settle(self._bill_estimates(models, call_id), quiet=True)
            raise
        for response in responses:
            response.usage.call_id = call_id
            self._apply_cast_json(response)
        await settle(self._bill_answers(responses), quiet=False)
        for response in responses:
            await self._fire(response, disable_usage=True)
        return [self._finalize(r) for r in responses]

    async def call_prefer(self, models: list[str], *, priority: int = 0) -> ResponseT:
        """Try models in order; advance to the next on failure or non-SUCCESS status.

        ``priority`` orders each attempt in the rate-limit queue (higher first); it is
        forwarded to every underlying ``call``.

        Fallback triggers (try next model):
        - ``LLMModelNotFoundError`` — model not configured in the gateway.
        - ``LLMServerError`` — 5xx response from the gateway.
        - ``LLMTimeoutError`` — HTTP-level timeout before the server replied.
        - Response with ``status != SUCCESS`` (server ran but reported failure).

        Immediate re-raise (not model-specific, no fallback):
        - ``LLMAuthError``, ``LLMCapabilityError``.

        A failed attempt bills what a plain ``call()`` bills, under its own call
        id: a timeout an estimate of its prompt (``estimated=True``, at the model's
        list price), a model-not-found, a 5xx or a failed connection nothing.
        When the server returns a non-SUCCESS response, ``on_usage`` was already
        called by ``call()`` with the server-reported usage; no duplicate call
        is made.

        Returns the last non-exception response (possibly non-SUCCESS) when all
        models are exhausted, or a ``NO_DEPLOYMENT`` response if ``models`` is
        empty or every attempt raised an exception.
        """
        _fallback_exc = (LLMModelNotFoundError, LLMServerError, LLMTimeoutError, LLMConnectionError)
        last: ResponseT | None = None
        for model in models:
            try:
                resp = await self.call(model, priority=priority)
                if resp.status == OutputStatus.SUCCESS:
                    return resp
                # Non-SUCCESS: on_usage already fired by call(); record and try next.
                last = resp
            except _fallback_exc as exc:
                # The attempt billed itself, as a plain call does: a timeout its estimate, anything else nothing.
                logger.warning("call_prefer: model=%s failed (%s), trying next", model, exc)
        if last is not None:
            return last
        return self._finalize(LLMResponse(model=models[-1] if models else "", status=OutputStatus.NO_DEPLOYMENT))

    def call_prefer_sync(self, models: list[str]) -> ResponseT:
        """Blocking wrapper for `call_prefer`."""
        return asyncio.run(self.call_prefer(models))

    async def _safe_call(self, model: str) -> ResponseT:
        """``call`` that converts model-specific failures into a ``NO_DEPLOYMENT`` result."""
        try:
            return await self.call(model)
        except (LLMModelNotFoundError, LLMServerError, LLMTimeoutError, LLMConnectionError) as exc:
            logger.warning("call_best: model=%s failed (%s), excluded", model, exc)
            return self._finalize(LLMResponse(model=model, status=OutputStatus.NO_DEPLOYMENT))

    async def _call_best_server(
        self, models: list[str], attribute: str, direction: Literal["greatest", "lowest"], timeout: float
    ) -> ResponseT:
        """Server-side best: one request fans out across `models`, returns the SUCCESS ranked best by `extra_attributes[attribute]`.

        It bills the one answer it returns. A request that times out, is cancelled or loses its
        connection before Gate answers bills one estimate of its prompt instead, priced as the first
        of ``models`` (the winner is unknown), then re-raises; one Gate never received or refused
        bills nothing.
        """
        await self._gate_budget()
        call_id = self._next_call_id()
        request = LLMRequest(
            target=BestTarget(models=models, attribute=attribute, direction=direction),
            call_id=call_id,
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
            hosting_providers=self.hosting_providers,
            plan=self.plan,
            operation=self.operation,
            deployment_id=self.deployment_id,
            no_fallback=self.no_fallback,
            fallback_on_content_policy=self.fallback_on_content_policy,
            seed_routing=self.seed_routing,
            **self._conversation_controls(),
        )
        try:
            with _naming_failures(call_id):
                response = await self.client._send(request)  # noqa: SLF001
        except (LLMTimeoutError, asyncio.CancelledError, *LOST_AFTER_SEND) as exc:
            # No answer, so no usage from Gate; a request that went out bills its prompt as an estimate.
            if not timed_out_unsent(exc):
                await settle(self._bill_estimates(models[:1], call_id), quiet=True)
            raise
        response.usage.call_id = call_id
        self._apply_cast_json(response)
        await settle(self._bill(response.usage), quiet=False)
        await self._fire(response, disable_usage=True)
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
        bills + fires callbacks via ``call``; a cancelled one bills an estimate of its prompt, its
        request having gone out. Returns a ``NO_DEPLOYMENT`` response if every model fails.

        ``greatest`` / ``lowest`` switch to SERVER-side selection: one request fans out across
        ``models`` and the gateway returns the SUCCESS whose model has the greatest/lowest
        ``extra_attributes[<name>]`` (no client-side weight fetching; ``key``/``accept`` ignored).
        """
        if greatest is not None and lowest is not None:
            msg = "Pass either `greatest` or `lowest`, not both."
            raise ValueError(msg)
        if not models:
            return self._finalize(LLMResponse(model="", status=OutputStatus.NO_DEPLOYMENT))
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
        return best[1] if best is not None else self._finalize(LLMResponse(model=models[-1], status=OutputStatus.NO_DEPLOYMENT))

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

    async def _call_tool_loop(self, model: str, *, disable_usage: bool = False) -> LLMResponse:
        """Non-streaming tool loop; returns the raw final turn with usage aggregated across turns."""
        executor = self.tool_executor
        if executor is None:
            return await self._call_raw(model, disable_usage=disable_usage)
        messages = list(self.messages)
        total = RawUsage(model=model, operation=self.operation)
        last: LLMResponse | None = None
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            # Each turn is its own Gate request: the caller's id (if any) goes to the first, the rest mint.
            turn = self.model_copy(
                update={
                    "messages": turn_messages,
                    "tool_executor": None,
                    "tools": self.tools if allow_tools else None,
                    "call_id": self._take_call_id(),
                }
            )
            response = await turn._call_raw(model, disable_usage=disable_usage)  # noqa: SLF001
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
        return LLMResponse(model=model, status=OutputStatus.NO_DEPLOYMENT)

    async def _stream_tool_loop(
        self,
        model: str,
        *,
        smooth: bool,
        server_side: bool,
        smooth_duration_ms: int,
        disable_usage: bool = False,
        usage_chunks: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming tool loop; yields each turn's chunks, runs tool calls between turns."""
        executor = self.tool_executor
        if executor is None:
            async for chunk in self.call_stream(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            ):
                yield chunk
            return
        messages = list(self.messages)
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            # Each turn is its own Gate request: the caller's id (if any) goes to the first, the rest mint.
            turn = self.model_copy(
                update={
                    "messages": turn_messages,
                    "tool_executor": None,
                    "tools": self.tools if allow_tools else None,
                    "call_id": self._take_call_id(),
                }
            )
            text_parts: list[str] = []
            acc: dict[int, ToolCall] = {}
            turn_stream = turn.call_stream(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            )
            # Closed with the loop, so a caller leaving mid-turn settles that turn's usage now.
            async with Closing(turn_stream) as chunks:
                async for chunk in chunks:
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
        disable_usage: bool = False,
        usage_chunks: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming tool loop whose executor yields live progress.

        Each turn streams the model's chunks, then runs the tools concurrently: ``ToolProgress``
        items are surfaced into the output stream as they arrive, and each tool's ``ToolResult``
        is fed back to the model. The loop repeats until a turn requests no tools, ``max_iters``
        is hit, or a tool returns ``ToolResult(redo=False)``.
        """
        executor = self.tool_stream_executor
        if executor is None:
            async for chunk in self.call_stream(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            ):
                yield chunk
            return
        messages = list(self.messages)
        tools = list(self.tools or [])
        for _ in range(self.tool_max_iters):
            allow_tools = self.tool_max_tokens_before_use is None or _conversation_token_count(messages) <= self.tool_max_tokens_before_use
            turn_messages = messages if allow_tools else [*messages, Message.user(_TOOL_BUDGET_EXCEEDED_NOTE)]
            turn = self.model_copy(
                update={
                    "messages": turn_messages,
                    "tool_executor": None,
                    "tool_stream_executor": None,
                    "tools": tools if allow_tools else None,
                    "call_id": self._take_call_id(),
                },
            )
            text_parts: list[str] = []
            acc: dict[int, ToolCall] = {}
            turn_stream = turn.call_stream(
                model,
                smooth=smooth,
                server_side=server_side,
                smooth_duration_ms=smooth_duration_ms,
                disable_usage=disable_usage,
                usage_chunks=usage_chunks,
            )
            # Closed with the loop, so a caller leaving mid-turn settles that turn's usage now.
            async with Closing(turn_stream) as chunks:
                async for chunk in chunks:
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
            try:
                while done < len(tasks):
                    tc, item = await queue.get()
                    if item is None:
                        done += 1
                    elif isinstance(item, ToolProgress):
                        yield StreamChunk(text=item.content)
                    else:
                        messages.append(Message.tool(tc.id, item.output or "", name=tc.function.name, images=item.images))
                        tools.extend(_unoffered_tools(tools, item.add_tools))
                        answered.add(tc.id)
                        if not item.redo:
                            retrigger = False
                await asyncio.gather(*tasks)
            finally:
                # A caller who left mid-tool (aclose() or a cancellation) stops the tools, and waits for them to unwind.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            # OpenAI requires a tool message per tool_call_id on the next turn; backfill any the tool left silent.
            messages.extend(Message.tool(tc.id, "", name=tc.function.name) for tc in tool_calls if tc.id not in answered)
            if not retrigger:
                return

    def _build_request(
        self, model: str, stream: bool, *, call_id: UUID, smooth_server_side: bool = False, smooth_duration_ms: int = 0
    ) -> LLMRequest:
        target = FallbackTarget(models=self.prefer_models) if self.prefer_models else SingleTarget(model=model)
        return LLMRequest(
            target=target,
            call_id=call_id,
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
            hosting_providers=self.hosting_providers,
            plan=self.plan,
            operation=self.operation,
            deployment_id=self.deployment_id,
            no_fallback=self.no_fallback,
            fallback_on_content_policy=self.fallback_on_content_policy,
            seed_routing=self.seed_routing,
            **self._conversation_controls(),
            cache_call=self.cache_call,
            cache_ttl=self.cache_ttl,
            smooth=smooth_server_side,
            smooth_duration_ms=smooth_duration_ms,
        )


class StreamRequestBuilder:
    """Streaming view of a chat request, returned by ``RequestBuilder.stream(...)``.

    Streaming is a request *mode*: the pacing / usage options are captured once here, and the
    model-selection terminals mirror the non-streaming builder — ``call(model)`` (one model,
    ``SingleTarget``) and ``call_prefer(models)`` (server-side ``FallbackTarget``) — but yield
    ``StreamChunk``s. Both delegate to ``RequestBuilder.call_stream``, so streaming and
    non-streaming share one prefer / tool-loop / deployment-rotation path.
    """

    def __init__(
        self,
        source: RequestBuilder[Any],
        *,
        smooth: bool,
        server_side: bool,
        smooth_duration_ms: int,
        priority: int,
        disable_usage: bool,
        usage_chunks: bool,
    ) -> None:
        """Capture the source builder and streaming options (see ``RequestBuilder.stream``)."""
        self._source = source
        self._smooth = smooth
        self._server_side = server_side
        self._smooth_duration_ms = smooth_duration_ms
        self._priority = priority
        self._disable_usage = disable_usage
        self._usage_chunks = usage_chunks

    def _run(self, model: str, prefer: list[str] | None) -> AsyncIterator[StreamChunk]:
        """Dispatch to ``call_stream`` with the captured options; ``prefer`` selects the target."""
        source = self._source
        if prefer:
            # The copy dispatches in the source's place, so it takes the caller's id with it.
            source = source.model_copy(update={"prefer_models": prefer, "call_id": source._take_call_id()})  # noqa: SLF001
        return source.call_stream(
            model,
            smooth=self._smooth,
            server_side=self._server_side,
            smooth_duration_ms=self._smooth_duration_ms,
            priority=self._priority,
            disable_usage=self._disable_usage,
            usage_chunks=self._usage_chunks,
        )

    async def call(self, model: str) -> AsyncIterator[StreamChunk]:
        """Stream one model (``SingleTarget``)."""
        async with Closing(self._run(model, None)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def call_prefer(self, models: list[str]) -> AsyncIterator[StreamChunk]:
        """Stream with server-side model fallback (``FallbackTarget``).

        The gateway tries ``models`` in order per inference and commits to the first that emits
        a token (preferring ``models[0]``) — one request, no client-side re-send. Reuses the
        same mechanism as a non-streamed ``call_prefer``; runs the tool loop when ``with_tools``
        is set, so every turn falls back independently.
        """
        if not models:
            return
        async with Closing(self._run(models[0], models)) as chunks:
            async for chunk in chunks:
                yield chunk


class JsonRequestBuilder[ResponseT: JsonLLMResponse](RequestBuilder[ResponseT]):
    """Intermediary tier: parses each result's ``raw_text`` into ``json_response``.

    ``.cast_json()`` yields this; ``.cast(T)`` (``CastedRequestBuilder``) extends it with a typed value.
    """

    def _parsed_json(self, response: LLMResponse) -> dict[str, Any] | None:
        """The effective JSON dict: server-set ``json_object`` if present, else parsed from ``raw_text``."""
        parsed = (
            response.json_object
            if response.json_object is not None
            else (_parse_json(response.raw_text, loose=self.loose_edges, repair=self.repair) if response.raw_text else None)
        )
        return cast("dict[str, Any] | None", parsed)

    def _finalize(self, response: LLMResponse) -> ResponseT:
        return JsonLLMResponse.of(response, self._parsed_json(response))  # type: ignore[return-value]  # ty: ignore[invalid-return-type]


class CastedRequestBuilder[T](JsonRequestBuilder[TypedLLMResponse[T]]):
    """Typed builder: a JSON request that also validates the parsed output into ``T`` (``.value``).

    ``T`` is a Pydantic model (``.cast(Model)`` — parses a JSON object) or ``list[Model]``
    (``.cast(list[Model])`` — parses a top-level JSON array, each element validated). Inherits
    every ``call`` / ``multicall`` / ``call_prefer`` / ``call_best`` variant and only overrides
    ``_finalize``. ``key`` / ``accept`` callbacks receive the typed result, so you can rank by ``.value``.
    """

    # A Pydantic model class, or a ``list[Model]`` alias (validated element-wise).
    model_type: Any

    def _finalize(self, response: LLMResponse) -> TypedLLMResponse[T]:
        if get_origin(self.model_type) is list:
            return self._finalize_list(response)
        parsed = self._parsed_json(response)
        value: Any = None
        if response.status == OutputStatus.SUCCESS and parsed is not None:
            try:
                value = self.model_type.model_validate(parsed)
            except ValidationError:
                value = None
        return TypedLLMResponse.of(response, parsed, value)

    def _finalize_list(self, response: LLMResponse) -> TypedLLMResponse[T]:
        value: Any = None
        if response.status == OutputStatus.SUCCESS:
            items = extract_json_list_from_text(response.raw_text, repair=self.repair)
            if items is not None:
                try:
                    # ``TypeAdapter`` validates the whole ``list[...]`` — element type may be a
                    # Pydantic model, a primitive (``list[int]``), or ``dict`` (``list[dict]``).
                    value = TypeAdapter(self.model_type).validate_python(items)
                except ValidationError:
                    value = None
        return TypedLLMResponse.of(response, None, value)


# ---------------------------------------------------------------------------
# Image / Audio / Video builders
# ---------------------------------------------------------------------------


class ImageRequestBuilder(MediaBuilder[ImageResponse]):
    """Fluent helper from ``LLMClient.image``; ``.call(model)`` / ``.call_stream(model)`` send it."""

    request: ImageRequest

    async def call(self, model: str) -> ImageResponse:
        """Generate or edit images with ``model`` (edit mode when ``images`` was set)."""
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        with _naming_failures(call_id):
            response = await self.client._send_images(request)  # noqa: SLF001
        response.usage.call_id = call_id
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> ImageResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def call_stream(self, model: str) -> AsyncIterator[ImageData]:
        """Stream partial image frames; after the final frame, fire usage + result callbacks."""
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        frames: list[ImageData] = []
        terminal_usage: RawUsage | None = None
        with _naming_failures(call_id):
            async for chunk in self.client._stream_images(request):  # noqa: SLF001
                if isinstance(chunk, ImageData):
                    frames.append(chunk)
                    yield chunk
                else:
                    terminal_usage = chunk
        usage = terminal_usage or RawUsage(model=request.model)
        usage.call_id = call_id
        synth = ImageResponse(model=request.model, data=frames, usage=usage)
        await self._fire(synth)

    def stream(self) -> ImageStreamBuilder:
        """Streaming view: ``.stream().call(model)`` / ``.stream().call_prefer(models)`` yield partial ``ImageData`` frames.

        Mirrors the chat builder's ``.stream()`` so an image request can be built once and then
        either streamed (progressive partial frames) or sent directly via ``.call``/``.call_prefer``.
        """
        return ImageStreamBuilder(self)


class ImageStreamBuilder:
    """Streaming view of an image request, returned by ``ImageRequestBuilder.stream()``.

    ``call(model)`` streams partial ``ImageData`` frames for one model. ``call_prefer(models)`` adds
    client-side fallback: it tries ``models`` in order but, since emitted frames cannot be un-yielded,
    commits to the first model that produces a frame — falling back only when a model fails *before*
    its first frame. The last pre-frame error is re-raised when every model fails.
    """

    def __init__(self, source: ImageRequestBuilder) -> None:
        """Capture the source image builder whose request/callbacks this streams."""
        self._source = source

    async def call(self, model: str) -> AsyncIterator[ImageData]:
        """Stream partial frames for a single ``model``."""
        async for frame in self._source.call_stream(model):
            yield frame

    async def call_prefer(self, models: list[str]) -> AsyncIterator[ImageData]:
        """Stream partial frames, committing to the first model that emits one (see class docstring)."""
        last_error: Exception | None = None
        for model in models:
            started = False
            try:
                async for frame in self._source.call_stream(model):
                    started = True
                    yield frame
            except (LLMModelNotFoundError, LLMServerError, LLMTimeoutError, LLMConnectionError, TimeoutError) as exc:
                if started:
                    raise
                logger.warning("image stream call_prefer: model=%s failed pre-frame (%s), trying next", model, exc)
                last_error = exc
                continue
            return
        if last_error is not None:
            raise last_error
        msg = "call_prefer requires at least one model."
        raise ValueError(msg)


class TTSRequestBuilder(MediaBuilder[BaseAudioResponse]):
    """Speech builder from ``LLMClient.audio(mode='speech')``; supports ``.call`` and ``.call_stream``."""

    request: TTSRequest

    async def call(self, model: str) -> BaseAudioResponse:
        """Synthesize speech with ``model`` (base64-encoded audio + usage)."""
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        with _naming_failures(call_id):
            response = await self.client._send_tts(request)  # noqa: SLF001
        response.usage.call_id = call_id
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> BaseAudioResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))

    async def call_stream(self, model: str) -> AsyncIterator[bytes]:
        """Stream raw speech bytes; fire the usage on every exit, and the result callbacks at the end.

        Gate's speech stream sends no usage frame and bills the text's characters at the model's
        input price, so the usage is that estimate (``input_tokens == len(text)``, ``estimated=True``),
        fired once by every exit of a stream Gate received: the end, the caller stopping
        (``aclose()``), a cancellation, a timeout, or an error after audio arrived. A stream Gate
        never received or refused fires nothing.
        """
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        delivered = False
        error: BaseException | None = None
        try:
            with _naming_failures(call_id):
                async with Closing(self.client._stream_tts(request)) as audio:  # noqa: SLF001
                    async for chunk in audio:
                        delivered = True
                        yield chunk
        except BaseException as exc:
            error = exc
            raise
        finally:
            if _speech_owes_usage(error, delivered=delivered):
                await settle(self._bill_speech(request, call_id, finished=error is None), quiet=error is not None)

    async def _bill_speech(self, request: TTSRequest, call_id: UUID, *, finished: bool) -> None:
        """Fire a speech stream's estimate; a ``finished`` stream also hands its result to the callbacks."""
        price = await self.client._list_price(request.model, request.plan)  # noqa: SLF001
        usage = estimated_usage(
            price, model=request.model, input_tokens=len(request.text), output_tokens=0, operation=request.operation, call_id=call_id
        )
        if not finished:
            await self._fire_usage(usage)
            return
        await self._fire(TTSResponse(model=request.model, audio="", response_format=request.response_format, usage=usage))


def _speech_owes_usage(error: BaseException | None, *, delivered: bool) -> bool:
    """Whether a speech stream leaving with ``error`` (``None`` for its end) owes a usage: Gate received it and did not refuse it."""
    if isinstance(error, LLMTimeoutError):
        return not timed_out_unsent(error)
    return error is None or isinstance(error, (GeneratorExit, asyncio.CancelledError)) or delivered


class AudioGenRequestBuilder(MediaBuilder[BaseAudioResponse]):
    """Generative-audio builder from ``LLMClient.audio(mode='music'|'sound_effects'|'dialogue')``."""

    request: AudioGenRequest

    async def call(self, model: str) -> BaseAudioResponse:
        """Generate music / sound effects / dialogue with ``model`` (base64-encoded audio + usage)."""
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        with _naming_failures(call_id):
            response = await self.client._send_audio_gen(request)  # noqa: SLF001
        response.usage.call_id = call_id
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> BaseAudioResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))


class VideoRequestBuilder(MediaBuilder[VideoResponse]):
    """Fluent helper from ``LLMClient.video``; ``.call(model)`` sends it (long-running, no stream)."""

    request: VideoRequest

    async def call(self, model: str) -> VideoResponse:
        """Generate a video with ``model`` and parse the response (base64-encoded video + usage)."""
        await self._gate_budget()
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        with _naming_failures(call_id):
            response = await self.client._send_video(request)  # noqa: SLF001
        response.usage.call_id = call_id
        await self._fire(response)
        return response

    def call_sync(self, model: str) -> VideoResponse:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))


class DirectRequestBuilder[ResponseT: LLMCallRecord](MediaBuilder[ResponseT]):
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
        call_id = self._next_call_id()
        request = self.request.model_copy(update={"model": model, "call_id": call_id})
        with _naming_failures(call_id):
            response = await self.client._post_json(self.path, request, self.label, client_timeout=self.client_timeout)  # noqa: SLF001
        parsed = self.response_type.model_validate(response.json())
        parsed.usage.call_id = call_id
        await self._fire(parsed)
        return parsed

    def call_sync(self, model: str) -> ResponseT:
        """Blocking wrapper for `call`."""
        return asyncio.run(self.call(model))
