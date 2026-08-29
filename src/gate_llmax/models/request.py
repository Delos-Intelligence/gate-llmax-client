"""Request models for the Gate LLM Gateway."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from ..types import JsonDict, ReasoningEffort
from .messages import Message


class CallControl(BaseModel):
    """Shared per-call retry/timeout overrides for chat, embed, and audio requests.

    ``max_tries`` / ``timeout`` fall back to the resolved model's defaults
    (``LLMModel.max_tries`` / ``LLMModel.timeout``) when left as ``None``.
    """

    max_tries: int | None = Field(default=None, description="Per-call upstream attempts, including the first.")
    timeout: int | None = Field(default=None, description="Per-call upstream timeout in seconds.")
    operation: str = Field(default="", description="Caller-supplied usage tag; echoed onto RawUsage and the usage log row.")
    deployment_id: str | None = Field(
        default=None,
        description="Dev keys only: serve from this deployment id alone, skipping filters, rotation and fallback.",
    )
    cache_call: bool | None = Field(
        default=None,
        description=(
            "Whether to serve this call from the gateway response cache. ``None`` defers to the "
            "API key's ``response_caching`` default; True/False force it on/off for this call."
        ),
    )
    cache_ttl: int | None = Field(
        default=None,
        description=(
            "Response-cache lifetime in seconds when caching is active. ``None`` uses the gateway "
            "default (600s). Has no effect on its own — caching is enabled by ``cache_call``."
        ),
    )


class ZoneSelection(BaseModel):
    """Optional deployment zone filter applied by the backend when routing.

    When set on a request, only deployments whose ``country`` or
    ``provider_region`` matches one of the given values are considered.
    Both lists are case-insensitive.  A deployment is kept if it satisfies
    *all* non-None criteria (AND logic between fields, OR logic within each
    list).
    """

    countries: list[str] | None = None
    regions: list[str] | None = None

    @classmethod
    def country(cls, value: str) -> ZoneSelection:
        """Filter to deployments in this country (case-insensitive match)."""
        return cls(countries=[value])

    @classmethod
    def zone(cls, value: str) -> ZoneSelection:
        """Filter to deployments in this provider region (``provider_region``)."""
        return cls(regions=[value])


class RequestSpecifics(BaseModel):
    """Optional LLM call parameters forwarded to the provider.

    All fields default to None so callers only specify what they need.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    seed_sampling: int | None = Field(
        default=None,
        description="Provider sampling seed for deterministic sampling (OpenAI/Azure); distinct from LLMRequest.seed_routing.",
    )
    n: int | None = Field(
        default=None,
        description="Number of completions to generate (OpenAI/Azure `n`). When >1, the extra texts land in LLMResponse.choices.",
    )
    reasoning: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    prompt_cache_key: str | None = Field(
        default=None,
        description=(
            "Provider prompt-cache routing hint (OpenAI/Azure). Requests sharing a key route to "
            "the same backend so a long stable prefix hits the prefix cache. Forwarded as-is; "
            "leave unset for short prompts (it adds routing constraint with nothing to cache)."
        ),
    )
    extra_body: JsonDict | None = Field(
        default=None,
        description=(
            "Per-request escape hatch merged on top of the model's `default_body` and "
            "the translated typed fields above. Use only when you need a raw provider-"
            "specific override that the typed `specifics` fields don't cover."
        ),
    )


class ResolveRequest(BaseModel):
    """Ask the gateway what a chat call to ``model`` would route to, without calling it."""

    model: str
    zone_selection: ZoneSelection | None = None
    hosting_providers: list[str] | None = Field(
        default=None,
        description=(
            "Restrict routing to deployments on these hosting providers; a canonical slug also "
            "admits its tier variants (e.g. 'azure' includes 'azure-cheap', never the reverse). "
            "None/[] = no filter."
        ),
    )
    plan: str | None = Field(
        default=None,
        description=(
            "Named preset of hosting providers (a cost/infra tier, e.g. 'omicron'). Equivalent to "
            "setting hosting_providers to the plan's set; an explicit hosting_providers wins. None = no plan."
        ),
    )
    seed_routing: str | None = Field(
        default=None,
        description="Pre-hashed routing token (as a real call would send); when set, the response pins the exact deployment.",
    )
    session_id: str | None = Field(default=None, description="Sticky-session key; used as the pin when seed_routing is absent.")


class SingleTarget(BaseModel):
    """Route to one model."""

    kind: Literal["single"] = "single"
    model: str


class ParallelTarget(BaseModel):
    """Fan out across N models concurrently; return one response per model in input order."""

    kind: Literal["parallel"] = "parallel"
    models: list[str] = Field(min_length=1)
    specifics_by_model: dict[str, RequestSpecifics] | None = Field(
        default=None,
        description="Per-model overrides for the shared `specifics`. Absent models use the shared value.",
    )


class FallbackTarget(BaseModel):
    """Try models in order, returning the first SUCCESS. Falls back on any non-success status."""

    kind: Literal["fallback"] = "fallback"
    models: list[str] = Field(min_length=1)


class BestTarget(BaseModel):
    """Fan out across N models, then return the SUCCESS ranked best by `extra_attributes[attribute]`."""

    kind: Literal["best"] = "best"
    models: list[str] = Field(min_length=1)
    attribute: str = Field(description="Model `extra_attributes` key to rank successes by.")
    direction: Literal["greatest", "lowest"] = Field(default="greatest", description="Pick the greatest or lowest attribute value.")
    specifics_by_model: dict[str, RequestSpecifics] | None = Field(
        default=None,
        description="Per-model overrides for the shared `specifics`. Absent models use the shared value.",
    )


ModelTarget = Annotated[SingleTarget | ParallelTarget | FallbackTarget | BestTarget, Field(discriminator="kind")]


class LLMRequest(CallControl):
    """Canonical request payload sent from client to backend gateway.

    The ``target`` field selects the dispatch strategy:
    - ``SingleTarget``: one model (the only kind that supports ``stream=True``).
    - ``ParallelTarget``: fan-out, gather all responses.
    - ``FallbackTarget``: try in order until one succeeds.
    """

    system_prompt: str | list[JsonDict] = Field(
        default="",
        description=(
            "System prompt. A plain string (one auto prompt-cache breakpoint for Anthropic), or a list "
            "of Anthropic-style content blocks carrying caller-placed `cache_control` for manual "
            "breakpoints. Non-Anthropic providers flatten the blocks to text."
        ),
    )
    messages: list[Message] = Field(default_factory=list)
    tools: list[JsonDict] | None = Field(
        default=None,
        description=(
            "OpenAI-shaped function/tool schemas offered to the model. A request-level field "
            "(part of the prompt), not a per-call tuning knob — distinct from `specifics`."
        ),
    )
    tool_choice: str | JsonDict | None = Field(
        default=None,
        description="'auto' | 'none' | 'required' | {'type':'function','function':{'name':...}}. Translated for Anthropic.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Allow more than one tool call per turn. False maps to Anthropic `disable_parallel_tool_use`.",
    )
    response_format: JsonDict | None = Field(
        default=None,
        description="OpenAI/Azure response_format (e.g. {'type':'json_object'}); set by client `.cast_json()`/`.cast()` to force JSON.",
    )
    images: list[str] = Field(default_factory=list, description="Base64-encoded images")
    images_alternative: str | None = Field(
        default=None,
        description=(
            "Text substituted for `images` when the target model lacks `supports_images`. "
            "When set, the gateway strips images and appends this text to the last user "
            "message instead of raising HTTP 422."
        ),
    )
    specifics: RequestSpecifics = Field(default_factory=RequestSpecifics)
    target: ModelTarget
    stream: bool = False
    zone_selection: ZoneSelection | None = None
    hosting_providers: list[str] | None = Field(
        default=None,
        description=(
            "Restrict routing to deployments served by these hosting providers "
            "(slugs, e.g. 'azure', 'aws-bedrock', 'scaleway'). A canonical slug also "
            "admits its tier variants ('azure' includes 'azure-cheap'); a variant slug "
            "admits only itself ('azure-cheap' never widens to plain 'azure'). None or "
            "[] applies no filter. Composes with zone_selection (AND)."
        ),
    )
    plan: str | None = Field(
        default=None,
        description=(
            "Named preset of hosting providers (a cost/infra tier, e.g. 'omicron'). Resolves to the "
            "plan's hosting_providers set server-side; an explicit hosting_providers takes precedence. "
            "None = no plan."
        ),
    )
    seed_routing: str | None = Field(
        default=None,
        description=(
            "Opaque pre-hashed routing key for deterministic deployment pinning. The "
            "client hashes the raw seed, so the gateway never sees the original value. "
            "Takes precedence over the X-Gate-Session-Id header."
        ),
    )
    smooth: bool = Field(
        default=False,
        description="When true, the gateway paces streamed text frames server-side (see smooth_duration_ms).",
    )
    smooth_duration_ms: int = Field(
        default=0,
        description="Per-text-frame delay in ms applied when smooth=True.",
    )
    batch_timeout: float | None = Field(
        default=None,
        description="Wall-clock budget for the whole call. Only used with parallel/fallback targets.",
    )

    @field_validator("tools")
    @classmethod
    def _dedupe_tool_names(cls, tools: list[JsonDict] | None) -> list[JsonDict] | None:
        """Drop duplicate tool schemas by function name, keeping the first occurrence.

        Bedrock rejects the entire request with ``400 tools: Tool names must be unique`` when the
        same function name appears twice, so a caller that assembles its tool list from several
        sources (base tools + lazily unlocked toolboxes) kills the run before its first turn.
        Enforcing uniqueness on the request model covers every dispatch path (single, parallel,
        fallback, best, streaming and the tool loop), which all serialize this same field.
        """
        if not tools:
            return tools
        unique: list[JsonDict] = []
        seen: set[str] = set()
        for tool in tools:
            fn = tool.get("function")
            raw = fn.get("name") if isinstance(fn, dict) else tool.get("name")
            name = raw if isinstance(raw, str) and raw else None
            if name is not None:
                if name in seen:
                    continue
                seen.add(name)
            unique.append(tool)
        return unique
