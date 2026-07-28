"""LLMClient – main entry point for the Gate Python SDK."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Literal, Self, overload

import httpx
from pydantic import BaseModel

from gate_llmax.models.audio import AudioRequest, AudioResponse
from gate_llmax.models.audio_gen import AudioGenMode, AudioGenRequest, AudioGenResponse, AudioMode, DialogueTurn
from gate_llmax.models.audio_isolation import AudioIsolationRequest, AudioIsolationResponse
from gate_llmax.models.config import ExtraAttributeName, ModelInfo, ModelPlanRow, PlanInfo, ResolveResponse
from gate_llmax.models.dubbing import DubbingRequest, DubbingResponse
from gate_llmax.models.embed import EmbedRequest, EmbedResponse
from gate_llmax.models.images import AspectRatio, ImageData, ImageQuality, ImageRequest, ImageResponse, ImageSize
from gate_llmax.models.messages import Message, MessageRole, TextMessage
from gate_llmax.models.request import LLMRequest, RequestSpecifics, ResolveRequest, ZoneSelection
from gate_llmax.models.response import (
    LLMCallRecord,
    LLMResponse,
    MulticallStreamFrame,
    RawUsage,
    StreamChunk,
    VisionLLMResponse,
)
from gate_llmax.models.responses import ResponsesRequest, ResponsesResponse
from gate_llmax.models.tts import TTSFormat, TTSRequest, TTSResponse
from gate_llmax.models.usage import (
    ApiKeyName,
    DeploymentInfo,
    ErrorReport,
    ErrorSample,
    LatencyRow,
    StatsRow,
    StoredPayload,
    TimeseriesPoint,
    UsageSample,
)
from gate_llmax.models.video import VideoAspectRatio, VideoDuration, VideoRequest, VideoResolution, VideoResponse
from gate_llmax.models.vision import VisionOCRRequest
from gate_llmax.ratelimit import RateLimit, RateLimiter
from gate_llmax.types import JsonDict, JsonValue, ReasoningEffort

from .exceptions import (
    LLMAuthError,
    LLMCapabilityError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMEscapeHatchWarning,
    LLMModelNotFoundError,
    LLMServerError,
    LLMTimeoutError,
)
from .request import (
    AudioGenRequestBuilder,
    BudgetCheck,
    DirectRequestBuilder,
    ImageRequestBuilder,
    OnUsage,
    RequestBuilder,
    TTSRequestBuilder,
    UsageCallback,
    VideoRequestBuilder,
)
from .streaming import StreamResponse

# Outermost rung of the timeout ladder: strictly above Gate's own ceiling, so the gateway fires first.
GATEWAY_MAX_BUDGET = 630.0
CLIENT_MARGIN = 30.0
DEFAULT_TIMEOUT = GATEWAY_MAX_BUDGET + CLIENT_MARGIN
MEDIA_CLIENT_TIMEOUT = 630.0  # dubbing/video are long-running; outlast the server-side wait
VERIFY_CLIENT_TIMEOUT = 900.0  # a capability verification is dozens of live probes per endpoint

CONNECT_TIMEOUT = 10.0
POOL_TIMEOUT = 10.0
# The SSE read budget is the gap between frames, not the whole stream.
STREAM_READ_TIMEOUT = GATEWAY_MAX_BUDGET + CLIENT_MARGIN
STREAM_MAX_RETRIES = 2  # resume attempts after a severed connection
STREAM_RETRY_BACKOFF = 0.5
# finish_reason when a drop could not be resumed: the answer above it is partial.
STREAM_INTERRUPTED = "interrupted"

# Timeouts and connect failures are handled separately, by their own exceptions.
TRANSIENT_STREAM_ERRORS = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)

logger = logging.getLogger(__name__)


class LLMClient:
    """HTTP client for the Gate gateway; use `.request(...).call(model)` and variants."""

    _api_key: str
    _base_url: str
    _timeout: float
    _cache_call: bool | None
    _cache_ttl: int | None
    _default_temperature: float | None
    _usage_callbacks: list[UsageCallback]
    _budget: BudgetCheck | None
    _default_zone_selection: ZoneSelection | None
    _default_hosting_providers: list[str] | None
    _default_plan: str | None
    _seed_routing_token: str | None
    _http: httpx.AsyncClient
    _stream_timeout: httpx.Timeout
    _owns_http: bool

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        stream_read_timeout: float = STREAM_READ_TIMEOUT,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
        temperature: float | None = None,
        usage_callback: UsageCallback | None = None,
        budget: BudgetCheck | None = None,
        default_zone_selection: ZoneSelection | None = None,
        default_hosting_providers: list[str] | None = None,
        default_plan: str | None = None,
        seed_routing: object | None = None,
        rate_limit: RateLimit | None = None,
        httpx_aclient: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client with API key and base URL.

        Args:
            api_key: Gate API key sent as ``X-Gate-Key`` on every request.
            base_url: Base URL of the Gate gateway (trailing slash stripped).
            timeout: Default request timeout in seconds, applied per request even when
                ``httpx_aclient`` is shared, so a pool default can never silently shorten it.
            stream_read_timeout: Max gap between SSE frames before the client gives up.
            cache_call: Default response-cache switch applied to every call. ``None`` (the
                default) defers to the API key's server-side ``response_caching`` default;
                ``True`` / ``False`` force caching on / off. Per-call arguments override it.
            cache_ttl: Default response-cache lifetime in seconds applied to every call when
                caching is active. ``None`` (the default) uses the gateway default (600s).
                Sets the lifetime only — caching is enabled by ``cache_call``. Per-call
                arguments override this default.
            temperature: Default sampling temperature seeded onto every ``.request(...)`` /
                ``.simple_request(...)`` call whose ``specifics`` does not set one. ``None`` (the
                default) leaves the provider default in place; bind e.g. ``0.0`` here once for
                deterministic output instead of repeating it per call. A per-call ``specifics``
                with an explicit ``temperature`` always wins.
            usage_callback: Default ``RawUsage`` counter applied to every call (builders and
                direct methods). Bind a user/org here once — e.g. a credit-deduction coroutine —
                instead of repeating ``.usage_callback(...)`` per request. Per-call
                ``.usage_callback(...)`` adds to (does not replace) this default.
            budget: Default pre-call gate applied to every call; a ``False`` result raises
                ``LLMBudgetError`` before dispatch. Per-call ``.budget(...)`` overrides it.
            default_zone_selection: Default zone filter for every ``.request(...)`` call
                (chat routing only); per-call ``.zone(...)`` overrides it.
            default_hosting_providers: Default hosting-provider allow-list for every
                ``.request(...)`` call (slugs, e.g. ``["azure", "aws-bedrock"]``; a canonical
                slug also admits its tier variants — ``azure`` includes ``azure-cheap``);
                per-call ``.hosting(...)`` overrides it. ``None`` applies no filter.
            default_plan: Default plan — a named hosting-provider preset (e.g. ``"omicron"``) —
                applied to every ``.request(...)`` call; resolves to the plan's hosting providers
                server-side, and an explicit ``.hosting(...)`` wins. ``None`` = no plan.
            seed_routing: Default deterministic-routing seed (e.g. ``(org_id, user_id)``) pinning a
                principal's calls to one deployment; per-call ``seed_routing=`` overrides it.
            rate_limit: Optional client-side throttle (concurrency / requests-per-min / tokens-per-min)
                applied to every chat call — replaces wrapping the client in a rate-limited subclass.
            httpx_aclient: Optional shared async HTTP client.  When provided the
                caller owns its lifecycle and ``close()`` will not close it.
                Useful to share a single connection pool across many ``LLMClient``
                instances and avoid the memory-leak behaviour of repeatedly
                creating/destroying ``httpx.AsyncClient`` objects.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_call = cache_call
        self._cache_ttl = cache_ttl
        self._default_temperature = temperature
        self._usage_callbacks = [usage_callback] if usage_callback is not None else []
        self._total_usage = 0.0
        # Always present — an unset RateLimit is a no-op, so call paths need no None-check.
        self._limiter = RateLimiter(rate_limit or RateLimit())
        self._budget = budget
        self._default_zone_selection = default_zone_selection
        self._default_hosting_providers = default_hosting_providers
        self._default_plan = default_plan
        self._seed_routing_token = seed_to_token(seed_routing)
        self._owns_http = httpx_aclient is None
        self._http = httpx_aclient or httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-Gate-Key": self._api_key},
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT, pool=POOL_TIMEOUT),
        )
        self._stream_timeout = httpx.Timeout(
            stream_read_timeout,
            connect=CONNECT_TIMEOUT,
            write=timeout,
            pool=POOL_TIMEOUT,
        )

    def clear_usage_callbacks(self) -> None:
        """Drop all usage callbacks so subsequent calls on this client are unbilled / untracked.

        Useful for free-model endpoints: build the client, then disable its billing hook.
        """
        self._usage_callbacks.clear()

    def add_usage_callback(self, *callbacks: UsageCallback) -> None:
        """Register extra ``RawUsage`` callbacks fired after every billed call on this client.

        Runtime equivalent of the constructor's ``usage_callback`` — append a hook (e.g. a per-task
        cost capture) without rebuilding the client. Pair with ``clear_usage_callbacks()`` to reset.
        """
        self._usage_callbacks.extend(callbacks)

    def get_usage_callbacks(self) -> list[UsageCallback]:
        """Return a shallow copy of the registered client-level usage callbacks (for inspection)."""
        return list(self._usage_callbacks)

    @property
    def total_usage(self) -> float:
        """Accumulated USD cost of every billed call on this client (the end-of-run total).

        Each completed call adds its ``RawUsage.total_cost``; calls made with ``disable_usage=True``
        are excluded. Mirrors the in-process library's per-run accumulator: bill once at the end of a
        run with ``min(client.total_usage, cap)`` instead of per call. Call ``reset_total_usage()`` to
        start a fresh window (not needed for a per-run client, which starts at ``0.0``).
        """
        return self._total_usage

    def reset_total_usage(self) -> None:
        """Reset the accumulated ``total_usage`` to ``0.0`` to start a new accounting window."""
        self._total_usage = 0.0

    def _record_usage(self, usage: RawUsage) -> None:
        """Add one call's cost to ``total_usage`` — invoked by every builder's usage fire."""
        self._total_usage += usage.total_cost

    def _resolve_cache_call(self, override: bool | None) -> bool | None:
        """Per-call ``cache_call`` wins when provided; else the client default (``None`` = the key's)."""
        return self._cache_call if override is None else override

    def _resolve_cache_ttl(self, override: int | None) -> int | None:
        """Per-call ``cache_ttl`` wins when provided; else the client default (``None`` = the gateway's)."""
        return self._cache_ttl if override is None else override

    def request(
        self,
        system_prompt: str | list[JsonDict] = "",
        messages: list[Message] | None = None,
        prompt: str | None = None,
        images: list[str] | None = None,
        images_alternative: str | None = None,
        specifics: RequestSpecifics | None = None,
        timeout: int | None = None,
        max_tries: int | None = None,
        on_usage: OnUsage | None = None,
        *,
        operation: str,
        seed_routing: object | None = None,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> RequestBuilder[LLMResponse]:
        """Build a request builder with the given parameters.

        Args:
            system_prompt: Optional system prompt.
            messages: Conversation history (mutually exclusive with ``prompt``).
            prompt: Shorthand for a single user turn; builds one ``Message.user(prompt)``.
            images: Base64-encoded images to attach to the last user message.
            images_alternative: Text the gateway substitutes for ``images`` when the
                target model lacks image support. Set when fanning out across mixed
                rosters so non-multimodal models get a textual proxy instead of HTTP 422.
            specifics: Optional LLM parameters (temperature, tools, …).
            timeout: Per-request timeout in seconds; overrides the client default.
            max_tries: Retry budget; overrides the model default.
            on_usage: Async coroutine called with ``RawUsage`` after each completed
                request (non-streaming) or after the final chunk (streaming).
            operation: Caller-supplied usage tag, echoed onto every ``RawUsage`` and
                the server-side usage log row.
            seed_routing: Any JSON-serializable value (e.g. ``(org_id, user_id)``) for
                deterministic deployment pinning. Hashed client-side to an opaque
                token, so the raw value never reaches the gateway. Overrides the
                client-level ``seed_routing`` default when provided.
            cache_call: Whether to serve this call from the gateway response cache;
                overrides the client default. ``None`` inherits it (and, with no client
                default, defers to the API key's server-side ``response_caching``).
            cache_ttl: Per-call response-cache lifetime in seconds when caching is active;
                overrides the client default. ``None`` inherits it; caching itself is
                enabled by ``cache_call``.
        """
        routing_token = seed_to_token(seed_routing) if seed_routing is not None else self._seed_routing_token
        if prompt is not None and messages is not None:
            msg = "Pass either `prompt` or `messages`, not both."
            raise ValueError(msg)
        if prompt is not None:
            if messages is not None:
                msg = "Pass either `prompt` or `messages`, not both."
                raise ValueError(msg)
            resolved_messages = [Message.user(prompt)]
        else:
            resolved_messages = messages or []
        resolved_specifics = specifics or RequestSpecifics()
        if resolved_specifics.temperature is None and self._default_temperature is not None:
            resolved_specifics = resolved_specifics.model_copy(update={"temperature": self._default_temperature})
        return RequestBuilder[LLMResponse](
            client=self,
            system_prompt=system_prompt,
            messages=resolved_messages,
            images=images or [],
            images_alternative=images_alternative,
            specifics=resolved_specifics,
            timeout=timeout,
            max_tries=max_tries,
            on_usage=on_usage,
            zone_selection=self._default_zone_selection,
            hosting_providers=self._default_hosting_providers,
            plan=self._default_plan,
            operation=operation,
            seed_routing=routing_token,
            cache_call=self._resolve_cache_call(cache_call),
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
            usage_callbacks=list(self._usage_callbacks),
            budget_check=self._budget,
        )

    def simple_request(
        self,
        prompt: str | list[Message] = "",
        *,
        system_prompt: str | list[JsonDict] = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        operation: str,
        max_tries: int | None = None,
        timeout: int | None = None,
        seed_routing: object | None = None,
        specifics: RequestSpecifics | None = None,
    ) -> RequestBuilder[LLMResponse]:
        """Terse ``.request(...)`` for the common case: flat tuning kwargs instead of a ``RequestSpecifics``.

        Returns a normal ``RequestBuilder`` — the model goes on the shared terminal (``.call`` /
        ``.call_prefer`` / ``.call_best``) and every fluent method still chains. ``prompt`` is a string
        (one user turn) or a ``Message`` list; pass ``specifics`` for non-tuning fields. For JSON, chain
        ``.cast_json()`` / ``.cast(T)``.
        """
        overrides = {
            k: v
            for k, v in {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "reasoning_effort": reasoning_effort}.items()
            if v is not None
        }
        return self.request(
            system_prompt=system_prompt,
            messages=prompt if isinstance(prompt, list) else None,
            prompt=prompt if isinstance(prompt, str) else None,
            specifics=(specifics or RequestSpecifics()).model_copy(update=overrides),
            operation=operation,
            max_tries=max_tries,
            timeout=timeout,
            seed_routing=seed_routing,
        )

    def _direct_builder[T: LLMCallRecord](
        self,
        request: BaseModel,
        path: str,
        label: str,
        response_type: type[T],
        *,
        client_timeout: float | None = None,
    ) -> DirectRequestBuilder[T]:
        """Build a `DirectRequestBuilder` seeded with the client-level usage/budget defaults."""
        return DirectRequestBuilder(
            client=self,
            request=request,
            path=path,
            label=label,
            response_type=response_type,
            client_timeout=client_timeout,
            usage_callbacks=list(self._usage_callbacks),
            budget_check=self._budget,
        )

    def embed(
        self,
        input: str | list[str],  # noqa: A002
        *,
        operation: str,
        max_tries: int | None = None,
        timeout: int | None = None,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> DirectRequestBuilder[EmbedResponse]:
        """Fluent builder for embeddings; ``.call(model)`` sends it.

        ``input`` is one or more strings to embed. ``operation`` tags the usage row.
        ``cache_call`` / ``cache_ttl`` / ``max_tries`` / ``timeout`` default to the client's.
        """
        request = EmbedRequest(
            model="",
            input=input,
            operation=operation,
            max_tries=max_tries,
            timeout=timeout,
            cache_call=self._resolve_cache_call(cache_call),
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return self._direct_builder(request, "/v1/embeddings", "Embedding", EmbedResponse)

    def transcribe(
        self,
        audio_b64: str,
        *,
        operation: str,
        language: str | None = None,
        response_format: str = "text",
        prompt: str | None = None,
        temperature: float | None = None,
        max_tries: int | None = None,
        timeout: int | None = None,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> DirectRequestBuilder[AudioResponse]:
        """Fluent builder for audio transcription; ``.call(model)`` sends it.

        ``audio_b64`` is base64 audio; ``operation`` tags the usage row; ``language`` is a BCP-47
        hint; ``response_format`` is one of ``text`` / ``json`` / ``verbose_json`` / ``srt`` / ``vtt``.
        ``prompt`` biases decoding (domain vocabulary, spelling); ``temperature`` sets the sampling temperature.
        """
        request = AudioRequest(
            model="",
            audio=audio_b64,
            operation=operation,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            max_tries=max_tries,
            timeout=timeout,
            cache_call=self._resolve_cache_call(cache_call),
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return self._direct_builder(request, "/v1/audio/transcriptions", "Audio transcription", AudioResponse)

    def isolation(
        self,
        audio_b64: str,
        *,
        operation: str,
        duration_seconds: float = 0.0,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[AudioIsolationResponse]:
        """Fluent builder for audio isolation; ``.call(model)`` sends it.

        ``audio_b64`` is the base64 source; ``operation`` tags the usage row;
        ``duration_seconds`` is used only for usage/billing.
        """
        request = AudioIsolationRequest(
            model="", audio=audio_b64, operation=operation, duration_seconds=duration_seconds, max_tries=max_tries, timeout=timeout
        )
        return self._direct_builder(request, "/v1/audio/isolation", "Audio isolation", AudioIsolationResponse)

    def dub(
        self,
        source_url: str,
        source_lang: str,
        target_lang: str,
        *,
        operation: str,
        duration_seconds: float = 0.0,
        watermark: bool = False,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[DubbingResponse]:
        """Fluent builder for dubbing; ``.call(model)`` sends it (long-running).

        ``source_url`` is publicly reachable media; ``operation`` tags the usage row;
        ``source_lang`` / ``target_lang`` are ISO-639-1 codes; ``duration_seconds`` is used
        only for usage/billing.
        """
        request = DubbingRequest(
            model="",
            source_url=source_url,
            source_lang=source_lang,
            target_lang=target_lang,
            operation=operation,
            duration_seconds=duration_seconds,
            watermark=watermark,
            max_tries=max_tries,
            timeout=timeout,
        )
        return self._direct_builder(request, "/v1/audio/dubbing", "Dubbing", DubbingResponse, client_timeout=MEDIA_CLIENT_TIMEOUT)

    @overload
    def audio(
        self,
        mode: Literal["speech"],
        *,
        text: str = ...,
        voice: str = ...,
        response_format: TTSFormat = ...,
        speed: float = ...,
        max_tries: int | None = ...,
        timeout: int | None = ...,
        operation: str,
        cache_call: bool | None = ...,
        cache_ttl: int | None = ...,
    ) -> TTSRequestBuilder: ...

    @overload
    def audio(
        self,
        mode: AudioGenMode,
        *,
        prompt: str = ...,
        inputs: list[DialogueTurn] | None = ...,
        music_length_ms: int = ...,
        force_instrumental: bool = ...,
        duration_seconds: float | None = ...,
        prompt_influence: float | None = ...,
        language_code: str | None = ...,
        output_format: str = ...,
        max_tries: int | None = ...,
        timeout: int | None = ...,
        operation: str,
        cache_call: bool | None = ...,
        cache_ttl: int | None = ...,
    ) -> AudioGenRequestBuilder: ...

    def audio(
        self,
        mode: AudioMode,
        *,
        text: str = "",
        voice: str = "alloy",
        response_format: TTSFormat = "mp3",
        speed: float = 1.0,
        prompt: str = "",
        inputs: list[DialogueTurn] | None = None,
        music_length_ms: int = 30000,
        force_instrumental: bool = False,
        duration_seconds: float | None = None,
        prompt_influence: float | None = None,
        language_code: str | None = None,
        output_format: str = "mp3_44100_128",
        max_tries: int | None = None,
        timeout: int | None = None,
        operation: str,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> TTSRequestBuilder | AudioGenRequestBuilder:
        """Fluent builder for audio; ``.call(model)`` sends it. ``mode`` picks speech vs generative.

        ``speech`` → TTS (``text`` / ``voice`` / ``response_format`` / ``speed``; also supports
        ``.call_stream``). ``music`` / ``sound_effects`` → ``prompt`` (+ length / duration knobs).
        ``dialogue`` → ``inputs`` speaker turns.
        """
        use_cache = self._resolve_cache_call(cache_call)
        ttl = self._resolve_cache_ttl(cache_ttl)
        if mode == "speech":
            request: TTSRequest | AudioGenRequest = TTSRequest(
                model="",
                text=text,
                voice=voice,
                response_format=response_format,
                speed=speed,
                max_tries=max_tries,
                timeout=timeout,
                operation=operation,
                cache_call=use_cache,
                cache_ttl=ttl,
            )
            return TTSRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)
        request = AudioGenRequest(
            model="",
            mode=mode,
            prompt=prompt,
            inputs=inputs or [],
            music_length_ms=music_length_ms,
            force_instrumental=force_instrumental,
            duration_seconds=duration_seconds,
            prompt_influence=prompt_influence,
            language_code=language_code,
            output_format=output_format,
            max_tries=max_tries,
            timeout=timeout,
            operation=operation,
            cache_call=use_cache,
            cache_ttl=ttl,
        )
        return AudioGenRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)

    def video(
        self,
        prompt: str,
        *,
        aspect_ratio: VideoAspectRatio = "16:9",
        duration_seconds: VideoDuration = 6,
        resolution: VideoResolution = "720p",
        with_audio: bool = False,
        reference_images: list[str] | None = None,
        start_image: str | None = None,
        end_image: str | None = None,
        max_tries: int | None = None,
        timeout: int | None = None,
        operation: str,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> VideoRequestBuilder:
        """Fluent builder for text/image-to-video; ``.call(model)`` sends it.

        Args:
            prompt: Text prompt (ignored when ``start_image`` drives image-to-video).
            aspect_ratio: ``16:9`` or ``9:16``.
            duration_seconds: Clip length (4, 6, or 8 seconds).
            resolution: ``720p``, ``1080p``, or ``4k``.
            with_audio: Whether to generate an audio track.
            reference_images: Base64 style/content references (ignored with start/end frames).
            start_image: Base64 first frame (image-to-video).
            end_image: Base64 last frame.
            max_tries: Per-call upstream attempts; overrides the model default.
            timeout: Per-call upstream timeout in seconds; overrides the model default.
            operation: Caller-supplied usage tag, echoed onto the usage log row.
            cache_call: Whether to use the gateway response cache; overrides the client
                default. ``None`` defers to the API key's server-side default.
            cache_ttl: Response-cache lifetime in seconds when caching is active; overrides
                the client default. Caching itself is enabled by ``cache_call``.
        """
        request = VideoRequest(
            model="",
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
            with_audio=with_audio,
            reference_images=reference_images,
            start_image=start_image,
            end_image=end_image,
            max_tries=max_tries,
            timeout=timeout,
            operation=operation,
            cache_call=self._resolve_cache_call(cache_call),
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return VideoRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)

    def responses_request(
        self,
        input: JsonValue,  # noqa: A002
        *,
        extra_body: JsonDict | None = None,
        operation: str,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[ResponsesResponse]:
        """Fluent builder proxying a raw OpenAI Responses call (escape hatch — discouraged).

        Prefer ``.request(...).call(...)`` / ``.with_tools(...)``: this bypasses Gate's typed
        request normalization, for Responses-API features the unified chat surface lacks.
        ``input`` is a string or list of input items; only ``input`` / ``extra_body`` are forwarded.
        """
        warnings.warn(
            "LLMClient.responses_request() is an escape hatch that bypasses Gate's unified request/usage "
            "handling; prefer .request(...).call(...) or .with_tools(...).",
            LLMEscapeHatchWarning,
            stacklevel=2,
        )
        request = ResponsesRequest(model="", input=input, extra_body=extra_body, operation=operation, max_tries=max_tries, timeout=timeout)
        client_timeout = float(timeout) + 30.0 if timeout is not None else None
        return self._direct_builder(request, "/v1/responses", "Responses", ResponsesResponse, client_timeout=client_timeout)

    def vision(
        self,
        images: list[str],
        *,
        operation: str,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[VisionLLMResponse]:
        """Fluent builder for vision OCR; ``.call(model)`` sends it. ``images`` are base64-encoded; ``operation`` tags the usage row."""
        request = VisionOCRRequest(model="", images=images, operation=operation, max_tries=max_tries, timeout=timeout)
        return self._direct_builder(request, "/v1/vision/ocr", "Vision OCR", VisionLLMResponse)

    def image(
        self,
        prompt: str,
        *,
        images: list[bytes] | None = None,
        b64_images: list[str] | None = None,
        mask: str | None = None,
        n: int = 1,
        quality: ImageQuality = "medium",
        size: ImageSize | None = (1024, 1024),
        aspect_ratio: AspectRatio | None = None,
        background: Literal["transparent", "opaque", "auto"] | None = None,
        output_format: Literal["png", "jpeg", "webp"] | None = None,
        output_compression: int | None = None,
        partial_images: int = 3,
        max_tries: int | None = None,
        timeout: int | None = None,
        operation: str,
        cache_call: bool | None = None,
        cache_ttl: int | None = None,
    ) -> ImageRequestBuilder:
        """Fluent builder for image generation / edit; ``.call(model)`` sends it.

        Pass input images as raw bytes (``images``) and/or already-encoded base64
        (``b64_images``); supplying either switches the call to edit mode. ``cache_call`` /
        ``cache_ttl`` / ``max_tries`` / ``timeout`` default to the client's; ``operation`` tags the usage row;
        the rest mirror ``ImageRequest``.
        """
        encoded = [base64.b64encode(b).decode() for b in images or []] + list(b64_images or [])
        request = ImageRequest(
            model="",
            prompt=prompt,
            images=encoded or None,
            mask=mask,
            n=n,
            quality=quality,
            size=size,
            aspect_ratio=aspect_ratio,
            background=background,
            output_format=output_format,
            output_compression=output_compression,
            partial_images=partial_images,
            max_tries=max_tries,
            timeout=timeout,
            operation=operation,
            cache_call=self._resolve_cache_call(cache_call),
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return ImageRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)

    async def list_models(self) -> list[ModelInfo]:
        """GET /v1/models — registered models (capabilities, pricing, extra_attributes)."""
        response = await self._http.get("/v1/models")
        _raise_for_status(response)
        return [ModelInfo.model_validate(item) for item in response.json()]

    async def list_extra_attributes(self) -> list[ExtraAttributeName]:
        """GET /v1/extra-attributes — the registered extra-attribute names."""
        response = await self._http.get("/v1/extra-attributes")
        _raise_for_status(response)
        return [ExtraAttributeName.model_validate(item) for item in response.json()]

    async def list_plans(self) -> list[PlanInfo]:
        """GET /v1/plans — the hosting plans (cost/infra tiers). Requires a ``dev`` API key (403 otherwise)."""
        response = await self._http.get("/v1/plans")
        _raise_for_status(response)
        return [PlanInfo.model_validate(item) for item in response.json()]

    async def verify_profile(
        self,
        model: str,
        *,
        only: list[str] | None = None,
        every_replica: bool = False,
        include_parked: bool = False,
    ) -> JsonDict:
        """POST /v1/verify/chat/{model} — what each endpoint really accepts, against what the catalog claims.

        Requires a ``dev`` API key (403 otherwise). Every probe is one live completion, so this spends money.
        """
        payload = {"only": only or [], "every_replica": every_replica, "include_parked": include_parked}
        try:
            response = await self._http.post(f"/v1/verify/chat/{model}", json=payload, timeout=VERIFY_CLIENT_TIMEOUT)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Verification of {model!r} timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc
        _raise_for_status(response)
        return response.json()

    async def verify_probes(self) -> list[JsonDict]:
        """GET /v1/verify/probes — the probe catalogue a verification runs. Requires a ``dev`` API key."""
        response = await self._http.get("/v1/verify/probes")
        _raise_for_status(response)
        return response.json()

    async def list_api_key_names(self) -> list[ApiKeyName]:
        """GET /v1/usage/keys — every API key name. Requires a ``dev`` API key (403 otherwise).

        The usage routes are addressed by key *name*, so this is how you learn the valid ones.
        """
        response = await self._http.get("/v1/usage/keys")
        _raise_for_status(response)
        return [ApiKeyName.model_validate(item) for item in response.json()]

    async def usage_errors(
        self,
        *,
        since: str = "24h",
        keys: list[str] | None = None,
        models: list[str] | None = None,
        operations: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> ErrorReport:
        """GET /v1/usage/errors — what failed in a window, grouped, worst first.

        ``since`` is a duration (``24h``, ``7d``, ``90m``, ``2w``) or an ISO timestamp.
        ``keys`` filters by API key *name*; an unknown name is a 404 rather than an empty
        result, so a typo can't read as "nothing failed". Requires a ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("limit", limit)]
        params += [("key", k) for k in keys or []]
        params += [("model", m) for m in models or []]
        params += [("operation", o) for o in operations or []]
        params += [("status", s) for s in statuses or []]
        response = await self._http.get("/v1/usage/errors", params=params)
        _raise_for_status(response)
        return ErrorReport.model_validate(response.json())

    async def usage_error_samples(
        self,
        *,
        since: str = "24h",
        keys: list[str] | None = None,
        statuses: list[str] | None = None,
        operations: list[str] | None = None,
        search: str | None = None,
        replayable_only: bool = False,
        limit: int = 20,
    ) -> list[ErrorSample]:
        """GET /v1/usage/error-samples — individual failed calls, newest first.

        ``search`` matches a substring of the provider's error message. Requires a ``dev`` key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("limit", limit)]
        params += [("key", k) for k in keys or []]
        params += [("status", s) for s in statuses or []]
        params += [("operation", o) for o in operations or []]
        if search:
            params.append(("search", search))
        if replayable_only:
            params.append(("replayable_only", "true"))
        response = await self._http.get("/v1/usage/error-samples", params=params)
        _raise_for_status(response)
        return [ErrorSample.model_validate(item) for item in response.json()]

    async def usage_payload(self, log_id: str) -> StoredPayload:
        """GET /v1/usage/payload/{log_id} — the request body behind a failed call.

        404 when nothing was stored: bodies are kept only for failures that reached a
        provider, so successes and routing refusals have nothing to replay. Requires a
        ``dev`` API key.
        """
        response = await self._http.get(f"/v1/usage/payload/{log_id}")
        _raise_for_status(response)
        return StoredPayload.model_validate(response.json())

    async def usage_latency(
        self,
        *,
        since: str = "7d",
        group: str = "model",
        models: list[str] | None = None,
        deployments: list[str] | None = None,
        hosting_providers: list[str] | None = None,
        statuses: list[str] | None = None,
        buckets: list[int] | None = None,
        min_calls: int = 1,
        limit: int = 50,
    ) -> list[LatencyRow]:
        """GET /v1/usage/latency — TTFT/duration percentiles and decode speed per model or deployment.

        ``buckets`` are input-token edges (``[2000, 20000]``) that split each group by prompt
        size — the way to tell queueing from prompt processing. Requires a ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("group", group), ("limit", limit)]
        params += [("model", m) for m in models or []]
        params += [("deployment", d) for d in deployments or []]
        params += [("hosting_provider", h) for h in hosting_providers or []]
        params += [("status", s) for s in statuses or []]
        if buckets:
            params.append(("buckets", ",".join(str(b) for b in buckets)))
        if min_calls > 1:
            params.append(("min_calls", min_calls))
        response = await self._http.get("/v1/usage/latency", params=params)
        _raise_for_status(response)
        return [LatencyRow.model_validate(item) for item in response.json()]

    async def usage_timeseries(
        self,
        *,
        since: str = "24h",
        interval: str = "1h",
        models: list[str] | None = None,
        deployments: list[str] | None = None,
        hosting_providers: list[str] | None = None,
        keys: list[str] | None = None,
    ) -> list[TimeseriesPoint]:
        """GET /v1/usage/timeseries — calls, failures, median TTFT and cost per time bucket.

        ``interval`` is the bucket width (``5m``, ``1h``, ``1d``); the window is capped at 500
        buckets. Requires a ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("interval", interval)]
        params += [("model", m) for m in models or []]
        params += [("deployment", d) for d in deployments or []]
        params += [("hosting_provider", h) for h in hosting_providers or []]
        params += [("key", k) for k in keys or []]
        response = await self._http.get("/v1/usage/timeseries", params=params)
        _raise_for_status(response)
        return [TimeseriesPoint.model_validate(item) for item in response.json()]

    async def usage_stats(
        self,
        *,
        since: str = "24h",
        group: str = "model",
        models: list[str] | None = None,
        keys: list[str] | None = None,
        operations: list[str] | None = None,
        limit: int = 50,
    ) -> list[StatsRow]:
        """GET /v1/usage/stats — volume, error rate and spend per model, key or operation.

        The denominators ``usage_errors`` lacks: total calls next to failures. Requires a
        ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("group", group), ("limit", limit)]
        params += [("model", m) for m in models or []]
        params += [("key", k) for k in keys or []]
        params += [("operation", o) for o in operations or []]
        response = await self._http.get("/v1/usage/stats", params=params)
        _raise_for_status(response)
        return [StatsRow.model_validate(item) for item in response.json()]

    async def list_deployments(
        self,
        *,
        models: list[str] | None = None,
        hosting_providers: list[str] | None = None,
        statuses: list[str] | None = None,
    ) -> list[DeploymentInfo]:
        """GET /v1/usage/deployments — deployment configuration + health by name; no ids, no secrets.

        Requires a ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = []
        params += [("model", m) for m in models or []]
        params += [("hosting_provider", h) for h in hosting_providers or []]
        params += [("status", s) for s in statuses or []]
        response = await self._http.get("/v1/usage/deployments", params=params)
        _raise_for_status(response)
        return [DeploymentInfo.model_validate(item) for item in response.json()]

    async def usage_samples(
        self,
        *,
        since: str = "24h",
        keys: list[str] | None = None,
        models: list[str] | None = None,
        deployments: list[str] | None = None,
        statuses: list[str] | None = None,
        operations: list[str] | None = None,
        search: str | None = None,
        min_ttft_ms: int | None = None,
        min_duration_ms: int | None = None,
        include_route: bool = False,
        include_preview: bool = False,
        replayable_only: bool = False,
        limit: int = 20,
    ) -> list[UsageSample]:
        """GET /v1/usage/samples — individual calls with timings, newest first; successes included on demand.

        ``usage_error_samples`` generalized: pass ``statuses=["SUCCESS"]`` with ``min_ttft_ms``
        to inspect slow successful calls, ``include_route=True`` for the retry/fallback trace.
        Requires a ``dev`` API key.
        """
        params: list[tuple[str, str | int | float | None]] = [("since", since), ("limit", limit)]
        params += [("key", k) for k in keys or []]
        params += [("model", m) for m in models or []]
        params += [("deployment", d) for d in deployments or []]
        params += [("status", s) for s in statuses or []]
        params += [("operation", o) for o in operations or []]
        if search:
            params.append(("search", search))
        if min_ttft_ms is not None:
            params.append(("min_ttft_ms", min_ttft_ms))
        if min_duration_ms is not None:
            params.append(("min_duration_ms", min_duration_ms))
        if include_route:
            params.append(("include_route", "true"))
        if include_preview:
            params.append(("include_preview", "true"))
        if replayable_only:
            params.append(("replayable_only", "true"))
        response = await self._http.get("/v1/usage/samples", params=params)
        _raise_for_status(response)
        return [UsageSample.model_validate(item) for item in response.json()]

    async def model_plan_matrix(self) -> list[ModelPlanRow]:
        """GET /v1/model-plan-matrix — every model and the plans it is reachable on.

        Requires a ``dev`` API key (403 otherwise). Use it to build a ``call_prefer([...])``
        list covering every plan an app serves (see the ``gate-llmax agent`` MCP ``prefer_list`` tool).
        """
        response = await self._http.get("/v1/model-plan-matrix")
        _raise_for_status(response)
        return [ModelPlanRow.model_validate(item) for item in response.json()]

    async def resolve(
        self,
        model: str,
        *,
        zone_selection: ZoneSelection | None = None,
        hosting_providers: list[str] | None = None,
        plan: str | None = None,
        seed_routing: object | None = None,
        session_id: str | None = None,
    ) -> ResolveResponse:
        """POST /v1/resolve — preview what a chat call to ``model`` would route to, without calling it.

        Returns the resolved model (after redirects), the candidate deployments, and — when a pin key
        is given — the exact deployment selected. ``zone_selection``/``seed_routing`` default to the
        client's (seed hashed as for a real call), so this previews what ``.call(model)`` would do here.
        """
        request = ResolveRequest(
            model=model,
            zone_selection=zone_selection if zone_selection is not None else self._default_zone_selection,
            hosting_providers=hosting_providers if hosting_providers is not None else self._default_hosting_providers,
            plan=plan if plan is not None else self._default_plan,
            seed_routing=seed_to_token(seed_routing) if seed_routing is not None else self._seed_routing_token,
            session_id=session_id,
        )
        response = await self._http.post("/v1/resolve", content=request.model_dump_json(), headers={"Content-Type": "application/json"})
        _raise_for_status(response)
        return ResolveResponse.model_validate(response.json())

    async def _post_json(
        self,
        path: str,
        request: BaseModel,
        label: str,
        client_timeout: float | None = None,
    ) -> httpx.Response:
        """POST a Pydantic request as JSON, mapping transport errors uniformly.

        Shared by the non-streaming media endpoints; ``label`` is woven into the
        raised ``LLMTimeoutError`` / ``LLMConnectionError`` messages.
        """
        try:
            response = await self._http.post(
                path,
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=client_timeout if client_timeout is not None else self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{label} request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc
        _raise_for_status(response)
        return response

    async def _send_audio_gen(self, request: AudioGenRequest) -> AudioGenResponse:
        """POST to /v1/audio/generations and return an ``AudioGenResponse`` (long-running)."""
        response = await self._post_json("/v1/audio/generations", request, "Audio generation", client_timeout=MEDIA_CLIENT_TIMEOUT)
        return AudioGenResponse.model_validate(response.json())

    async def _send(self, request: LLMRequest, *, priority: int = 0) -> LLMResponse:
        """POST to /v1/chat/completions, throttled by the client's rate limiter (no-op when unset)."""
        async with self._limiter.guard(priority):
            try:
                response = await self._http.post(
                    "/v1/chat/completions",
                    content=request.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                    # Explicit: a shared httpx_aclient must not decide this client's ceiling.
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"Client-side ceiling ({self._timeout:.0f}s) hit before the gateway answered: {exc}") from exc
            except httpx.ConnectError as exc:
                raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc

            _raise_for_status(response)
            result = LLMResponse.model_validate(response.json())
        self._limiter.record_tokens(result.usage.input_tokens + result.usage.output_tokens)
        return result

    async def _send_batch(self, request: LLMRequest) -> list[LLMResponse]:
        """POST to /v1/chat/completions with a parallel target — single wire payload, N responses."""
        # Server-side batch wall-clock is `batch_timeout`; client-side httpx timeout must also accommodate it.
        client_timeout = (request.batch_timeout or 180.0) + 30.0
        try:
            response = await self._http.post(
                "/v1/chat/completions",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=client_timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Batch request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc

        _raise_for_status(response)
        return [LLMResponse.model_validate(item) for item in response.json()]

    async def _stream_batch(self, request: LLMRequest) -> AsyncIterator[MulticallStreamFrame]:
        """POST with stream=True + ParallelTarget; yield one frame per model completion (in completion order)."""
        client_timeout = (request.batch_timeout or 180.0) + 30.0
        try:
            async with self._http.stream(
                "POST",
                "/v1/chat/completions",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=client_timeout,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped or not stripped.startswith("data: "):
                        continue
                    payload = stripped[len("data: ") :]
                    if payload == "[DONE]":
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    try:
                        yield MulticallStreamFrame.model_validate(data)
                    except ValueError:
                        continue
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Multicall stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc

    async def _stream_once(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """One POST with stream=True — the wire leg the retry loop in ``_stream`` drives."""
        async with self._http.stream(
            "POST",
            "/v1/chat/completions",
            content=request.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=self._stream_timeout,
        ) as response:
            if response.status_code >= 400:
                await response.aread()
            _raise_for_status(response)
            async for chunk in StreamResponse(response):
                yield chunk

    async def _stream(self, request: LLMRequest, *, priority: int = 0) -> AsyncIterator[StreamChunk]:
        """POST with stream=True, throttled by the client's rate limiter (no-op when unset).

        A severed connection restarts here rather than reaching the caller, resuming from the
        text already delivered. Once the attempts are spent the stream ends on a chunk carrying
        ``finish_reason="interrupted"`` instead of raising.
        """
        async with self._limiter.guard(priority):
            total = 0
            delivered = ""  # text already handed to the caller; the resume point
            tool_started = False
            attempt = 0
            while True:
                try:
                    async for chunk in self._stream_once(resume_request(request, delivered) if delivered else request):
                        if chunk.output_tokens is not None:
                            total = (chunk.input_tokens or 0) + (chunk.output_tokens or 0)
                        if chunk.tool_calls_delta:
                            tool_started = True
                        delivered += chunk.text
                        yield chunk
                    break
                except httpx.TimeoutException as exc:
                    gap = self._stream_timeout.read or 0.0
                    msg = f"Client-side stream ceiling ({gap:.0f}s frame gap) hit before the gateway sent a frame: {exc}"
                    raise LLMTimeoutError(msg) from exc
                except httpx.ConnectError as exc:
                    raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc
                except TRANSIENT_STREAM_ERRORS as exc:
                    # A half-streamed tool call has truncated arguments; re-asking would emit
                    # a second call the caller would merge into the first.
                    if tool_started or attempt >= STREAM_MAX_RETRIES:
                        logger.exception("stream: dropped after %d chars, giving up", len(delivered))
                        yield StreamChunk(is_done=True, finish_reason=STREAM_INTERRUPTED)
                        break
                    delay = STREAM_RETRY_BACKOFF * 2**attempt
                    attempt += 1
                    logger.warning("stream: dropped after %d chars (%r); resuming in %.1fs (try %d)", len(delivered), exc, delay, attempt)
                    await asyncio.sleep(delay)
        self._limiter.record_tokens(total)

    async def _send_video(self, request: VideoRequest) -> VideoResponse:
        """POST to /v1/videos and return a ``VideoResponse`` (long-running)."""
        response = await self._post_json("/v1/videos", request, "Video generation", client_timeout=MEDIA_CLIENT_TIMEOUT)
        return VideoResponse.model_validate(response.json())

    async def _send_images(self, request: ImageRequest) -> ImageResponse:
        """POST to /v1/images and return an ``ImageResponse``."""
        try:
            response = await self._http.post(
                "/v1/images",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Images request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc
        _raise_for_status(response)
        return ImageResponse.model_validate(response.json())

    async def _stream_images(self, request: ImageRequest) -> AsyncIterator[ImageData | RawUsage]:
        """POST to /v1/images/stream and yield ``ImageData`` frames, then one terminal ``RawUsage``."""
        try:
            async with self._http.stream(
                "POST",
                "/v1/images/stream",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :].strip()
                    if payload == "[DONE]" or not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and "usage" in data:
                        try:
                            yield RawUsage.model_validate(data["usage"])
                        except ValueError:
                            continue
                        continue
                    try:
                        yield ImageData.model_validate(data)
                    except ValueError:
                        continue
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Images stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc

    async def _send_tts(self, request: TTSRequest) -> TTSResponse:
        """POST to /v1/audio/speech and return a ``TTSResponse``."""
        try:
            response = await self._http.post(
                "/v1/audio/speech",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"TTS request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc
        _raise_for_status(response)
        return TTSResponse.model_validate(response.json())

    async def _stream_tts(self, request: TTSRequest) -> AsyncIterator[bytes]:
        """POST to /v1/audio/speech/stream and yield raw audio bytes as they arrive."""
        try:
            async with self._http.stream(
                "POST",
                "/v1/audio/speech/stream",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                _raise_for_status(response)
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"TTS stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise LLMConnectionError(f"Could not connect to gateway: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying HTTP client.

        No-op when the client was supplied externally via ``httpx_aclient``.
        """
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self: Self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager."""
        await self.close()


RequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})
ImageRequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})
TTSRequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})
AudioGenRequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})
VideoRequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})
DirectRequestBuilder.model_rebuild(_types_namespace={"LLMClient": LLMClient})


def resume_request(request: LLMRequest, delivered: str) -> LLMRequest:
    """The same request with ``delivered`` fed back as an assistant turn, so the model continues it.

    Built from the original messages, so repeated resumes append one turn rather than a chain.
    """
    resumed = Message(role=MessageRole.ASSISTANT, content=[TextMessage(text=delivered)])
    return request.model_copy(update={"messages": [*request.messages, resumed]})


def seed_to_token(seed: object | None) -> str | None:
    """Hash a seed to an opaque token for deterministic routing (``None`` passes through)."""
    if seed is None:
        return None
    return hashlib.sha256(json.dumps(seed, sort_keys=True, default=str).encode()).hexdigest()


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP error codes to Gate exceptions."""
    code = response.status_code
    if code == 401:
        raise LLMAuthError("Invalid or missing API key")
    if code == 404:
        raise LLMModelNotFoundError(_extract_detail(response.text))
    if code == 400:
        raise LLMContentFilterError(_extract_detail(response.text))
    if code == 422:
        raise LLMCapabilityError(_extract_detail(response.text))
    if code >= 500:
        raise LLMServerError(f"Gateway server error {code}: {_extract_detail(response.text)}")
    if code >= 400:
        response.raise_for_status()


def _extract_detail(text: str) -> str:
    """Extract the ``detail`` field from a JSON error body, or return the raw text."""
    try:
        return json.loads(text).get("detail", text)
    except (json.JSONDecodeError, AttributeError):
        return text
