"""GateClient – main entry point for the Gate Python SDK."""

from __future__ import annotations

import base64
import hashlib
import json
import warnings
from collections.abc import AsyncIterator
from typing import Literal, Self, overload

import httpx
from pydantic import BaseModel

from gate_common.models.audio import AudioRequest, AudioResponse
from gate_common.models.audio_gen import AudioGenMode, AudioGenRequest, AudioGenResponse, AudioMode, DialogueTurn
from gate_common.models.audio_isolation import AudioIsolationRequest, AudioIsolationResponse
from gate_common.models.config import ExtraAttributeName, ModelInfo, ResolveResponse
from gate_common.models.dubbing import DubbingRequest, DubbingResponse
from gate_common.models.embed import EmbedRequest, EmbedResponse
from gate_common.models.images import AspectRatio, ImageData, ImageQuality, ImageRequest, ImageResponse, ImageSize
from gate_common.models.messages import Message
from gate_common.models.request import GateRequest, RequestSpecifics, ResolveRequest, ZoneSelection
from gate_common.models.response import (
    GateCallRecord,
    GateResponse,
    MulticallStreamFrame,
    RawUsage,
    StreamChunk,
    VisionGateResponse,
)
from gate_common.models.responses import ResponsesRequest, ResponsesResponse
from gate_common.models.tts import TTSFormat, TTSRequest, TTSResponse
from gate_common.models.video import VideoAspectRatio, VideoDuration, VideoRequest, VideoResolution, VideoResponse
from gate_common.models.vision import VisionOCRRequest
from gate_common.types import JsonDict, JsonValue, ReasoningEffort

from .exceptions import (
    GateAuthError,
    GateCapabilityError,
    GateConnectionError,
    GateEscapeHatchWarning,
    GateModelNotFoundError,
    GateServerError,
    GateTimeoutError,
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

_DEFAULT_TIMEOUT = 120.0
MEDIA_CLIENT_TIMEOUT = 630.0  # dubbing/video are long-running; outlast the server-side wait


class GateClient:
    """HTTP client for the Gate gateway; use `.request(...).call(model)` and variants."""

    _api_key: str
    _base_url: str
    _timeout: float
    _cache_ttl: int | None
    _default_temperature: float | None
    _usage_callbacks: list[UsageCallback]
    _budget: BudgetCheck | None
    _default_zone_selection: ZoneSelection | None
    _seed_routing_token: str | None
    _http: httpx.AsyncClient
    _owns_http: bool

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT,
        cache_ttl: int | None = None,
        temperature: float | None = None,
        usage_callback: UsageCallback | None = None,
        budget: BudgetCheck | None = None,
        default_zone_selection: ZoneSelection | None = None,
        seed_routing: object | None = None,
        httpx_aclient: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the client with API key and base URL.

        Args:
            api_key: Gate API key sent as ``X-Gate-Key`` on every request.
            base_url: Base URL of the Gate gateway (trailing slash stripped).
            timeout: Default request timeout in seconds.
            cache_ttl: Default response-cache lifetime in seconds applied to every call.
                ``None`` (the default) disables caching; a positive value caches successful
                responses for that many seconds. Per-call arguments override this default —
                pass ``0`` on a call to force caching off when the client default is on.
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
                ``GateBudgetError`` before dispatch. Per-call ``.budget(...)`` overrides it.
            default_zone_selection: Default zone filter for every ``.request(...)`` call
                (chat routing only); per-call ``.zone(...)`` overrides it.
            seed_routing: Default deterministic-routing seed (e.g. ``(org_id, user_id)``) pinning a
                principal's calls to one deployment; per-call ``seed_routing=`` overrides it.
            httpx_aclient: Optional shared async HTTP client.  When provided the
                caller owns its lifecycle and ``close()`` will not close it.
                Useful to share a single connection pool across many ``GateClient``
                instances and avoid the memory-leak behaviour of repeatedly
                creating/destroying ``httpx.AsyncClient`` objects.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._default_temperature = temperature
        self._usage_callbacks = [usage_callback] if usage_callback is not None else []
        self._budget = budget
        self._default_zone_selection = default_zone_selection
        self._seed_routing_token = seed_to_token(seed_routing)
        self._owns_http = httpx_aclient is None
        self._http = httpx_aclient or httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-Gate-Key": self._api_key},
            timeout=timeout,
        )

    def _resolve_cache_ttl(self, override: int | None) -> int | None:
        """Per-call ``cache_ttl`` wins when provided (``0`` forces off); else the client default."""
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
        operation: str = "",
        seed_routing: object | None = None,
        cache_ttl: int | None = None,
    ) -> RequestBuilder[GateResponse]:
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
            cache_ttl: Per-call response-cache lifetime in seconds; overrides the client
                default. ``None`` inherits the client default, ``0`` forces caching off,
                a positive value caches the successful response for that many seconds.
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
        return RequestBuilder[GateResponse](
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
            operation=operation,
            seed_routing=routing_token,
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
        operation: str = "",
        max_tries: int | None = None,
        timeout: int | None = None,
        seed_routing: object | None = None,
        specifics: RequestSpecifics | None = None,
    ) -> RequestBuilder[GateResponse]:
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

    def _direct_builder[T: GateCallRecord](
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

    def embed_request(
        self,
        input: str | list[str],  # noqa: A002
        *,
        max_tries: int | None = None,
        timeout: int | None = None,
        cache_ttl: int | None = None,
    ) -> DirectRequestBuilder[EmbedResponse]:
        """Fluent builder for embeddings; ``.call(model)`` sends it.

        ``input`` is one or more strings to embed. ``cache_ttl`` / ``max_tries`` / ``timeout``
        default to the client's.
        """
        request = EmbedRequest(model="", input=input, max_tries=max_tries, timeout=timeout, cache_ttl=self._resolve_cache_ttl(cache_ttl))
        return self._direct_builder(request, "/v1/embeddings", "Embedding", EmbedResponse)

    def transcribe_request(
        self,
        audio_b64: str,
        *,
        language: str | None = None,
        response_format: str = "text",
        prompt: str | None = None,
        temperature: float | None = None,
        max_tries: int | None = None,
        timeout: int | None = None,
        cache_ttl: int | None = None,
    ) -> DirectRequestBuilder[AudioResponse]:
        """Fluent builder for audio transcription; ``.call(model)`` sends it.

        ``audio_b64`` is base64 audio; ``language`` is a BCP-47 hint; ``response_format`` is one
        of ``text`` / ``json`` / ``verbose_json`` / ``srt`` / ``vtt``. ``prompt`` biases decoding
        (domain vocabulary, spelling); ``temperature`` sets the sampling temperature.
        """
        request = AudioRequest(
            model="",
            audio=audio_b64,
            language=language,
            response_format=response_format,
            prompt=prompt,
            temperature=temperature,
            max_tries=max_tries,
            timeout=timeout,
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return self._direct_builder(request, "/v1/audio/transcriptions", "Audio transcription", AudioResponse)

    def isolation_request(
        self,
        audio_b64: str,
        *,
        duration_seconds: float = 0.0,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[AudioIsolationResponse]:
        """Fluent builder for audio isolation; ``.call(model)`` sends it.

        ``audio_b64`` is the base64 source; ``duration_seconds`` is used only for usage/billing.
        """
        request = AudioIsolationRequest(model="", audio=audio_b64, duration_seconds=duration_seconds, max_tries=max_tries, timeout=timeout)
        return self._direct_builder(request, "/v1/audio/isolation", "Audio isolation", AudioIsolationResponse)

    def dub_request(
        self,
        source_url: str,
        source_lang: str,
        target_lang: str,
        *,
        duration_seconds: float = 0.0,
        watermark: bool = False,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[DubbingResponse]:
        """Fluent builder for dubbing; ``.call(model)`` sends it (long-running).

        ``source_url`` is publicly reachable media; ``source_lang`` / ``target_lang`` are
        ISO-639-1 codes; ``duration_seconds`` is used only for usage/billing.
        """
        request = DubbingRequest(
            model="",
            source_url=source_url,
            source_lang=source_lang,
            target_lang=target_lang,
            duration_seconds=duration_seconds,
            watermark=watermark,
            max_tries=max_tries,
            timeout=timeout,
        )
        return self._direct_builder(request, "/v1/audio/dubbing", "Dubbing", DubbingResponse, client_timeout=MEDIA_CLIENT_TIMEOUT)

    @overload
    def audio_request(
        self,
        mode: Literal["speech"],
        *,
        text: str = ...,
        voice: str = ...,
        response_format: TTSFormat = ...,
        speed: float = ...,
        max_tries: int | None = ...,
        timeout: int | None = ...,
        cache_ttl: int | None = ...,
    ) -> TTSRequestBuilder: ...

    @overload
    def audio_request(
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
        cache_ttl: int | None = ...,
    ) -> AudioGenRequestBuilder: ...

    def audio_request(
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
        cache_ttl: int | None = None,
    ) -> TTSRequestBuilder | AudioGenRequestBuilder:
        """Fluent builder for audio; ``.call(model)`` sends it. ``mode`` picks speech vs generative.

        ``speech`` → TTS (``text`` / ``voice`` / ``response_format`` / ``speed``; also supports
        ``.call_stream``). ``music`` / ``sound_effects`` → ``prompt`` (+ length / duration knobs).
        ``dialogue`` → ``inputs`` speaker turns.
        """
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
            cache_ttl=ttl,
        )
        return AudioGenRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)

    def video_request(
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
            cache_ttl: Response-cache lifetime in seconds; overrides the client default.
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
            cache_ttl=self._resolve_cache_ttl(cache_ttl),
        )
        return VideoRequestBuilder(client=self, request=request, usage_callbacks=list(self._usage_callbacks), budget_check=self._budget)

    def responses_request(
        self,
        input: JsonValue,  # noqa: A002
        *,
        extra_body: JsonDict | None = None,
        operation: str = "",
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[ResponsesResponse]:
        """Fluent builder proxying a raw OpenAI Responses call (escape hatch — discouraged).

        Prefer ``.request(...).call(...)`` / ``.with_tools(...)``: this bypasses Gate's typed
        request normalization, for Responses-API features the unified chat surface lacks.
        ``input`` is a string or list of input items; only ``input`` / ``extra_body`` are forwarded.
        """
        warnings.warn(
            "GateClient.responses_request() is an escape hatch that bypasses Gate's unified request/usage "
            "handling; prefer .request(...).call(...) or .with_tools(...).",
            GateEscapeHatchWarning,
            stacklevel=2,
        )
        request = ResponsesRequest(model="", input=input, extra_body=extra_body, operation=operation, max_tries=max_tries, timeout=timeout)
        client_timeout = float(timeout) + 30.0 if timeout is not None else None
        return self._direct_builder(request, "/v1/responses", "Responses", ResponsesResponse, client_timeout=client_timeout)

    def vision_request(
        self,
        images: list[str],
        *,
        max_tries: int | None = None,
        timeout: int | None = None,
    ) -> DirectRequestBuilder[VisionGateResponse]:
        """Fluent builder for vision OCR; ``.call(model)`` sends it. ``images`` are base64-encoded."""
        request = VisionOCRRequest(model="", images=images, max_tries=max_tries, timeout=timeout)
        return self._direct_builder(request, "/v1/vision/ocr", "Vision OCR", VisionGateResponse)

    def image_request(
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
        cache_ttl: int | None = None,
    ) -> ImageRequestBuilder:
        """Fluent builder for image generation / edit; ``.call(model)`` sends it.

        Pass input images as raw bytes (``images``) and/or already-encoded base64
        (``b64_images``); supplying either switches the call to edit mode. ``cache_ttl`` /
        ``max_tries`` / ``timeout`` default to the client's; the rest mirror ``ImageRequest``.
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

    async def resolve(
        self,
        model: str,
        *,
        zone_selection: ZoneSelection | None = None,
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
        raised ``GateTimeoutError`` / ``GateConnectionError`` messages.
        """
        try:
            response = await self._http.post(
                path,
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=client_timeout if client_timeout is not None else self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise GateTimeoutError(f"{label} request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc
        _raise_for_status(response)
        return response

    async def _send_audio_gen(self, request: AudioGenRequest) -> AudioGenResponse:
        """POST to /v1/audio/generations and return an ``AudioGenResponse`` (long-running)."""
        response = await self._post_json("/v1/audio/generations", request, "Audio generation", client_timeout=MEDIA_CLIENT_TIMEOUT)
        return AudioGenResponse.model_validate(response.json())

    async def _send(self, request: GateRequest) -> GateResponse:
        """POST to /v1/chat/completions and return a ``GateResponse``."""
        try:
            response = await self._http.post(
                "/v1/chat/completions",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise GateTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

        _raise_for_status(response)
        return GateResponse.model_validate(response.json())

    async def _send_batch(self, request: GateRequest) -> list[GateResponse]:
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
            raise GateTimeoutError(f"Batch request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

        _raise_for_status(response)
        return [GateResponse.model_validate(item) for item in response.json()]

    async def _stream_batch(self, request: GateRequest) -> AsyncIterator[MulticallStreamFrame]:
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
            raise GateTimeoutError(f"Multicall stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

    async def _stream(self, request: GateRequest) -> AsyncIterator[StreamChunk]:
        """POST with stream=True and yield StreamChunks via SSE."""
        try:
            async with self._http.stream(
                "POST",
                "/v1/chat/completions",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                _raise_for_status(response)
                stream = StreamResponse(response)
                async for chunk in stream:
                    yield chunk
        except httpx.TimeoutException as exc:
            raise GateTimeoutError(f"Stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

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
            raise GateTimeoutError(f"Images request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc
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
            raise GateTimeoutError(f"Images stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

    async def _send_tts(self, request: TTSRequest) -> TTSResponse:
        """POST to /v1/audio/speech and return a ``TTSResponse``."""
        try:
            response = await self._http.post(
                "/v1/audio/speech",
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise GateTimeoutError(f"TTS request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc
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
            raise GateTimeoutError(f"TTS stream timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GateConnectionError(f"Could not connect to gateway: {exc}") from exc

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


RequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})
ImageRequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})
TTSRequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})
AudioGenRequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})
VideoRequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})
DirectRequestBuilder.model_rebuild(_types_namespace={"GateClient": GateClient})


def seed_to_token(seed: object | None) -> str | None:
    """Hash a seed to an opaque token for deterministic routing (``None`` passes through)."""
    if seed is None:
        return None
    return hashlib.sha256(json.dumps(seed, sort_keys=True, default=str).encode()).hexdigest()


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP error codes to Gate exceptions."""
    code = response.status_code
    if code == 401:
        raise GateAuthError("Invalid or missing API key")
    if code == 404:
        raise GateModelNotFoundError(_extract_detail(response.text))
    if code == 422:
        raise GateCapabilityError(_extract_detail(response.text))
    if code >= 500:
        raise GateServerError(f"Gateway server error {code}: {_extract_detail(response.text)}")
    if code >= 400:
        response.raise_for_status()


def _extract_detail(text: str) -> str:
    """Extract the ``detail`` field from a JSON error body, or return the raw text."""
    try:
        return json.loads(text).get("detail", text)
    except (json.JSONDecodeError, AttributeError):
        return text
