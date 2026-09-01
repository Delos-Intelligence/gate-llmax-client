"""Gate MCP server — the minimum an agent needs to use Gate correctly.

Two kinds of tools, and the difference matters:

* **Free** — gateway metadata, no model ever runs, safe to call as often as you like:
  ``ping``, ``list_models``, ``list_plans``, ``model_plan_matrix``, ``plan_models``,
  ``model_in_plan``, ``prefer_list``, ``resolve``, ``heavy_test_cases``, ``verify_probes``,
  ``list_api_keys``, ``usage_errors``, ``usage_error_samples``, ``get_request_payload``,
  ``usage_latency``, ``usage_timeseries``, ``usage_stats``, ``usage_redirects``,
  ``fallback_health``, ``list_deployments``, ``usage_samples``.
* **Spends real money and quota** — ``heavy_test`` and ``verify_profile``. The first hammers a
  model with the shapes it claims to serve; the second probes whether those claims are right.
  Never reach for either to check the gateway is up or that a model exists — that is what
  ``ping`` and ``resolve`` are for.

Config comes from the environment, falling back to the file ``gate-llmax agent install`` writes:
    GATE_BASE_URL   base URL of the Gate gateway (e.g. https://gate.example.com)
    GATE_API_KEY    a Gate API key. Plan tools need a **dev** key (the `dev` flag); model/resolve
                    tools work with any key.

Run it with ``gate-llmax agent mcp`` (stdio). Requires the ``agent`` extra: ``pip install
"gate-llmax[agent]"`` / ``uv add "gate-llmax[agent]"``.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from gate_llmax.agent import credentials, lookup
from gate_llmax.agent.heavy_test import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_N,
    DEFAULT_OPERATION,
    DEFAULT_RATE_PER_MIN,
    case_catalogue,
    run_heavy_test,
)
from gate_llmax.agent.prefer import build_prefer_list
from gate_llmax.client import LLMClient
from gate_llmax.exceptions import LLMAuthError, LLMError
from gate_llmax.models.config import ModelInfo, ResolvedDeployment

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The Gate MCP server needs the 'agent' extra. Install it with:\n"
        '    uv add "gate-llmax[agent]"   (or)   pip install "gate-llmax[agent]"'
    ) from exc

mcp = FastMCP("gate-llmax")

_DEV_HINT = "This tool needs a dev API key. Set GATE_API_KEY to a key whose `dev` flag is true."


def _config() -> tuple[str, str]:
    base_url, api_key = credentials.resolve()
    if not base_url or not api_key:
        missing = ", ".join(n for n, v in (("GATE_BASE_URL", base_url), ("GATE_API_KEY", api_key)) if not v)
        raise RuntimeError(f"Missing gateway config: {missing}. Run `gate-llmax agent install` to set it.")
    return base_url, api_key


def _client() -> LLMClient:
    base_url, api_key = _config()
    return LLMClient(api_key=api_key, base_url=base_url)


def _dev_error(exc: Exception) -> str:
    """Turn an auth/permission failure into an actionable message for the agent."""
    text = str(exc)
    if isinstance(exc, LLMAuthError) or "403" in text or "dev" in text.lower():
        return f"{_DEV_HINT} (gateway said: {text})"
    return f"Gate request failed: {text}"


@mcp.tool()
async def ping() -> dict[str, Any]:
    """Free. Check the gateway is reachable and report which URL, key and plan access this server has.

    The first thing to call when something looks wrong: round-trip to the gateway, how many models
    it serves, and whether the configured key carries the ``dev`` flag the plan tools need. No
    model is called, so it costs nothing.
    """
    import time

    try:
        base_url, api_key = _config()
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    started = time.perf_counter()
    try:
        async with _client() as client:
            models = await client.list_models()
    except Exception as exc:  # any failure is the answer here, not an error to raise
        return {"ok": False, "base_url": base_url, "error": str(exc)}
    latency_ms = round((time.perf_counter() - started) * 1000)

    dev_key, dev_error = True, None
    try:
        async with _client() as client:
            await client.list_plans()
    except LLMError as exc:
        dev_key, dev_error = False, str(exc)

    return {
        "ok": True,
        "base_url": base_url,
        "api_key": f"…{api_key[-4:]}",
        "latency_ms": latency_ms,
        "models": len(models),
        "dev_key": dev_key,
        "dev_key_error": dev_error,
    }


@mcp.tool()
async def list_models(purpose: str | None = None, search: str | None = None) -> dict[str, Any]:
    """Free. Model names on the gateway, grouped by purpose then developer. Names only — call ``resolve`` for one model's details.

    Args:
        purpose: keep one purpose (chat, embed, audio, vision, images, tts, audio_isolation, dubbing, video).
        search: keep names matching these words, closest first (``opus``, ``gemini flash``).

    Works with any API key; no model is called.
    """
    try:
        async with _client() as client:
            models = await client.list_models()
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    if purpose:
        models = [m for m in models if m.purpose.value == purpose]
    if search:
        kept = set(lookup.search(search, [m.name for m in models]))
        models = [m for m in models if m.name in kept]
    grouped: dict[str, dict[str, list[str]]] = {}
    for model in models:
        developers = grouped.setdefault(model.purpose.value, {})
        developers.setdefault(lookup.developer_of(model.name, model.developer_id), []).append(model.name)
    for developers in grouped.values():
        for names in developers.values():
            names.sort()
    return {"count": len(models), "models": {p: dict(sorted(d.items())) for p, d in sorted(grouped.items())}}


@mcp.tool()
async def list_plans() -> list[dict[str, Any]]:
    """Free. List the gateway's hosting plans (cost/infra tiers), ordered by ``sort_order``.

    A plan is a named preset of hosting providers. Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            plans = await client.list_plans()
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [p.model_dump(mode="json") for p in plans]


@mcp.tool()
async def model_plan_matrix(purpose: str | None = None) -> list[dict[str, Any]]:
    """Free. For every model, the plans it is reachable on (has a deployment on an admitted host).

    Pass ``purpose`` to filter. Needs a **dev** API key. This is the raw data behind
    ``plan_models``, ``model_in_plan`` and ``prefer_list``.
    """
    try:
        async with _client() as client:
            rows = await client.model_plan_matrix()
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    out = [r.model_dump(mode="json", exclude={"model_id"}) for r in rows]
    if purpose:
        out = [r for r in out if r.get("purpose") == purpose]
    return out


@mcp.tool()
async def plan_models(plan_id: str, purpose: str | None = None) -> dict[str, Any]:
    """Free. List the models reachable on a given plan (optionally filtered by ``purpose``).

    Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            rows = await client.model_plan_matrix()
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    models = [r.model_name for r in rows if plan_id in r.available_plan_ids and (purpose is None or r.purpose.value == purpose)]
    return {"plan_id": plan_id, "purpose": purpose, "count": len(models), "models": sorted(models)}


@mcp.tool()
async def model_in_plan(model: str, plan_id: str) -> dict[str, Any]:
    """Free. Answer: is ``model`` reachable on ``plan_id``? Returns the model's full plan set too.

    Needs a **dev** API key. ``model`` is matched loosely, as in ``resolve``.
    """
    try:
        async with _client() as client:
            rows = await client.model_plan_matrix()
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    found = lookup.pick(model, [r.model_name for r in rows])
    row = next((r for r in rows if r.model_name == found.name), None)
    if row is None:
        return {"model": model, "plan_id": plan_id, "found": False, "available": False, "closest": found.candidates}
    return {
        "model": row.model_name,
        "plan_id": plan_id,
        "found": True,
        "available": plan_id in row.available_plan_ids,
        "available_plan_ids": row.available_plan_ids,
    }


@mcp.tool()
async def prefer_list(
    purpose: str = "chat",
    plans: list[str] | None = None,
    prefer: list[str] | None = None,
    operation: str = "my-operation",
) -> dict[str, Any]:
    """Free. Build a ``call_prefer([...])`` fallback list that covers every plan an app serves.

    Use this whenever the app targets cosmos-style plans. The returned ``models`` list, passed to
    ``.call_prefer(models)``, tries the best models first and always reaches a model reachable on
    the caller's plan (unless a plan is in ``uncovered_plans``).

    Args:
        purpose: model purpose to build for (default ``chat``).
        plans: plan ids to cover, in fallback-tail priority order. Omit to cover every plan; pass
            the ids from ``list_plans`` (already sorted) to control ordering.
        prefer: model names to try first, in quality order. Coverage is still guaranteed after them.
        operation: operation tag used in the generated snippet.

    Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            matrix = await client.model_plan_matrix()
            if plans is None:
                plans = [p.id for p in await client.list_plans()]
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    result = build_prefer_list(matrix, purpose=purpose, plans=plans or None, prefer=prefer)
    return {
        "purpose": purpose,
        "models": result.models,
        "coverage": result.coverage,
        "uncovered_plans": result.uncovered_plans,
        "steps": [{"model": s.model, "covers": s.covers} for s in result.steps],
        "snippet": result.snippet(operation),
    }


def _deployment_line(dep: ResolvedDeployment) -> str:
    """One deployment as a single line: name, host, region, priority, status, and the id ``heavy_test(deployment=)`` takes."""
    where = "/".join(p for p in (dep.hosting_provider, dep.region, dep.country) if p)
    parts = [dep.name, f"@ {where}" if where else "", f"({dep.api_provider})", f"p{dep.priority}", dep.status, f"id={dep.id}"]
    line = " ".join(p for p in parts if p)
    return f"{line} — {dep.last_error}" if dep.last_error else line


def _model_details(info: ModelInfo | None) -> dict[str, Any]:
    """Prices, capabilities and dates of a model, with empty fields left out."""
    if info is None:
        return {}
    supports = [k.removeprefix("supports_") for k, v in info.capabilities.model_dump().items() if v]
    out: dict[str, Any] = {
        "developer": lookup.developer_of(info.name, info.developer_id),
        "supports": supports,
        "usd_per_mtok": {"in": info.input_token_price, "out": info.output_token_price, "cached_in": info.input_cache_price},
        "max_tries": info.max_tries,
        "timeout": info.timeout,
        "created": lookup.short_date(info.created_at),
        "updated": lookup.short_date(info.updated_at),
    }
    if info.max_output_tokens:
        out["max_output_tokens"] = info.max_output_tokens
    if info.extra_attributes:
        out["extra_attributes"] = info.extra_attributes
    return {k: v for k, v in out.items() if v is not None}


@mcp.tool()
async def resolve(model: str, plan: str | None = None) -> dict[str, Any]:
    """Free. What a call to ``model`` would route to, plus that model's prices and capabilities. The way to look a model up.

    ``model`` need not be exact: ``opus 5`` finds ``claude-5-opus``. Too vague a query returns the
    three closest names instead of a resolution. Nothing is called, so this is free; any API key works.

    Args:
        model: registered name, alias, or a few words of one.
        plan: hosting plan to route under — deployments outside it are dropped from ``deployments``.
    """
    try:
        async with _client() as client:
            catalogue = await client.list_models()
            found = lookup.pick(model, [m.name for m in catalogue])
            if found.name is None and found.ambiguous:
                return {"query": model, "ambiguous": found.candidates, "hint": "several models match — resolve one of these names"}
            if found.name is None:
                return {"query": model, "found": False, "closest": found.candidates}
            resolved = await client.resolve(found.name, plan=plan)
    except LLMError as exc:
        return {"error": _dev_error(exc)}

    info = next((m for m in catalogue if m.name == resolved.resolved_model), None)
    out: dict[str, Any] = {"model": resolved.resolved_model, "purpose": resolved.purpose.value}
    if found.name and found.name.lower() != model.lower():
        out["matched_query"] = model
    if resolved.redirect_from:
        out["alias_of"] = resolved.redirect_from
    out |= _model_details(info)
    out["routing"] = resolved.selection_strategy
    out["deployments"] = [_deployment_line(d) for d in resolved.candidates]
    unroutable = [_deployment_line(d) for d in resolved.all_deployments if d.status != "ACTIVE"]
    if unroutable:
        out["unroutable"] = unroutable
    if resolved.fallbacks:
        out["fallbacks"] = " > ".join(r.model if r.deployment_count else f"{r.model}(no route)" for r in resolved.fallbacks)
    if not resolved.candidates:
        out["warning"] = "no routable deployment — a call to this model would fail"
    return out


@mcp.tool()
async def heavy_test_cases() -> list[dict[str, Any]]:
    """Free. List the request shapes ``heavy_test`` can run (id, intent, tags, required capabilities).

    Use it to pick a ``only=[...]`` subset before spending money on a full run. No gateway call.
    """
    return case_catalogue()


@mcp.tool()
async def heavy_test(
    model: str,
    n: int = DEFAULT_N,
    rate: float = DEFAULT_RATE_PER_MIN,
    plan: str | None = None,
    deployment: str | None = None,
    only: list[str] | None = None,
    operation: str = DEFAULT_OPERATION,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    *,
    include_runs: bool = False,
) -> dict[str, Any]:
    """SPENDS REAL MONEY. Hammer one chat model with every request shape it can serve, report what broke.

    This is the only tool here that calls a model. Use it deliberately — qualifying a model
    before it goes in front of users, or proving a suspected regression. To check the gateway is
    up use ``ping``; to check a model exists and has a deployment use ``resolve``; to see what the
    suite would run use ``heavy_test_cases``. None of those cost anything. For a cheap first
    signal, run the single smoke case once: ``heavy_test(model, n=1, only=["smoke"])``.

    ``deployment`` (a uuid from ``resolve``) pins every request in the suite to one endpoint,
    routable even while it is INACTIVE and with the fallback chain switched off — that is how a
    freshly added deployment is qualified before it takes production rotation.

    The suite is capability-matched: a model with ``supports_images`` gets the vision cases, one
    with ``supports_tools`` the tool-call cases, and so on (``heavy_test_cases`` lists them all),
    so a multimodal model is genuinely tested harder than a text-only one. Each answer is judged
    against declarative expectations — did the tool call fire, did the JSON parse, did the stop
    sequence hold, did the stream truncate with ``finish_reason=length``.

    **This spends real money and real quota**: it issues ``n x len(selected cases)`` requests
    (~20 cases for a fully capable model, so n=5 is ~100 calls) and takes roughly
    ``total / rate`` minutes. Launches are paced, not serialized: one request goes out every
    ``60 / rate`` seconds whatever the previous one is doing, so slow answers overlap the way
    production traffic does.

    Args:
        model: chat model name as registered on the gateway (e.g. ``gpt-5.6-terra``).
        n: how many times to replay the whole suite. Default 5.
        rate: launch rate in requests per minute. Default 6 (one every 10 seconds).
        plan: optional hosting plan to route under; ignored when ``deployment`` is set.
        deployment: deployment id (uuid, from ``resolve``) to pin every request to. Needs a **dev** key.
        only: restrict to these case ids or tags (e.g. ``["tools"]``, ``["smoke", "streaming"]``).
        operation: usage tag written to the gateway usage log.
        max_concurrency: ceiling on in-flight requests.
        budget_seconds: stop launching new requests after this much wall-clock (in-flight ones finish).
        include_runs: return every individual run record, not just the summary and the failures.

    Returns:
        Pass rate and status breakdown, latency and TTFT percentiles, token/cost totals, a
        per-case table, which deployments served the traffic, a determinism check, and the
        failing runs with their reasons. Works with any API key; ``deployment`` needs a dev one.
        With ``deployment`` set, ``deployment_pin`` says whether the pin actually held.
    """
    try:
        async with _client() as client:
            return await run_heavy_test(
                client,
                model,
                n=n,
                rate=rate,
                plan=plan,
                deployment=deployment,
                only=only,
                operation=operation,
                max_concurrency=max_concurrency,
                budget_seconds=budget_seconds,
                include_runs=include_runs,
            )
    except LLMError as exc:
        return {"error": _dev_error(exc)}


def _verify_digest(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep the features whose verdict is not plain ``ok`` — the rest is noise once a run passes."""
    return {
        "model": profile.get("model"),
        "declared": profile.get("declared"),
        "endpoints": [f"{e['hosting_provider']} ({e['name']})" for e in profile.get("endpoints", [])],
        "verdict_counts": dict(Counter(dim["verdict"] for dim in profile.get("dimensions", []))),
        "needs_attention": [dim for dim in profile.get("dimensions", []) if dim.get("verdict") != "ok"],
        "proposals": profile.get("proposals"),
        "code_findings": profile.get("code_findings"),
        "cost_usd": profile.get("cost_usd"),
        "duration_s": profile.get("duration_s"),
    }


@mcp.tool()
async def verify_probes() -> list[dict[str, Any]]:
    """Free. List the capability probes ``verify_profile`` runs (id, dimension, intent, the flag each speaks to).

    Use it to pick an ``only=[...]`` subset before spending money. Needs a dev key but calls no model.
    """
    try:
        async with _client() as client:
            return await client.verify_probes()
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]


@mcp.tool()
async def verify_profile(
    model: str,
    only: list[str] | None = None,
    *,
    every_replica: bool = False,
    include_parked: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    """SPENDS REAL MONEY. Probe what a chat model's endpoints really accept, and diff it against the catalog.

    Answers "do we restrict this model too much, or not enough?". For every feature — images,
    multi-image, tools, forced/parallel tool calls, tools+reasoning, each reasoning effort level,
    JSON object/schema mode, temperature, top_p, stop, n, seed, penalties, logprobs,
    prompt_cache_key, truncation — it works out what Gate would do to the request, then finds out
    whether it had to:

    - ``ok`` — Gate forwards it and the endpoint honours it.
    - ``over_restricted`` — Gate narrows or blocks it, but the endpoint accepts it.
    - ``under_restricted`` / ``not_honoured`` — Gate forwards it, the endpoint refuses or ignores it.
    - ``restriction_ok`` — Gate narrows it and the endpoint refuses it too.
    - ``unverifiable`` — Gate narrows it and this adapter offers no way to send it anyway.

    Restrictions living in the catalog (``supports_images`` / ``supports_tools`` /
    ``supports_reasoning`` / ``reasoning_efforts``) come back as ``proposals``: exact column writes
    an admin can apply from the dashboard. Restrictions living in Gate's own code come back as
    ``code_findings`` instead — no catalog write fixes those, they need a patch. This tool never
    writes anything.

    **Costs real money**: one live completion per probe per endpoint (~25 probes, so a few tenths
    of a cent and about a minute per hosting provider). ``verify_probes`` lists them for free.

    Args:
        model: chat model name as registered on the gateway.
        only: restrict to these probe ids or dimensions (e.g. ``["images", "reasoning_efforts"]``).
        every_replica: probe every deployment instead of one per hosting provider.
        include_parked: probe non-ACTIVE deployments too.
        full: return every probe run, not just the features needing attention.

    Returns:
        The per-feature verdicts that are not ``ok``, the proposed catalog writes, the code-level
        findings, and what the run cost. Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            profile = await client.verify_profile(model, only=only, every_replica=every_replica, include_parked=include_parked)
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    return profile if full else _verify_digest(profile)


# ---------------------------------------------------------------------------
# Usage insights — what is failing on the gateway, and how to reproduce it.
# All four need a **dev** API key, and all four see every key, not just this one.
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_api_keys() -> list[dict[str, Any]]:
    """Free. List the gateway's API key names (never the secrets).

    The usage tools below are addressed by key *name* — ``key="k8s cosmos prod"`` — so start
    here when you do not already know the exact name. Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            return [k.model_dump(mode="json") for k in await client.list_api_key_names()]
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]


@mcp.tool()
async def usage_errors(
    since: str = "24h",
    keys: list[str] | None = None,
    models: list[str] | None = None,
    operations: list[str] | None = None,
    statuses: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Free. What failed on the gateway in a window, grouped and worst first.

    The first thing to reach for when an app reports errors: it answers "how many, of what
    kind, on which model and which key", and each group carries the most recent provider
    message, so a count becomes a diagnosis in one call. No model runs; metadata only.

    Args:
        since: Duration (``1h``, ``24h``, ``7d``, ``2w``, ``90m``) or an ISO timestamp.
            Maximum 90 days.
        keys: API key **names** to restrict to (see ``list_api_keys``). Omit for all keys.
            An unknown name is an error, not an empty result — a typo cannot read as
            "nothing failed on that key".
        models: Model names to keep.
        operations: ``operation`` tags to keep (the caller-supplied label on each call).
        statuses: OutputStatus values to keep, e.g. ``["TIMEOUT", "RATE_LIMIT"]``.
        limit: Maximum groups returned (they are sorted worst first, so the head is the
            part that matters).

    Returns:
        The window, ``total_failures``, a ``by_status`` histogram, and ``groups`` — each with
        calls, cost, first/last seen, a ``sample_detail`` message, how many are ``replayable``
        and a ``sample_log_id`` to hand to ``get_request_payload``. Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            report = await client.usage_errors(
                since=since,
                keys=keys,
                models=models,
                operations=operations,
                statuses=statuses,
                limit=limit,
            )
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    return report.model_dump(mode="json")


@mcp.tool()
async def usage_error_samples(
    since: str = "24h",
    keys: list[str] | None = None,
    statuses: list[str] | None = None,
    operations: list[str] | None = None,
    search: str | None = None,
    limit: int = 20,
    *,
    replayable_only: bool = False,
) -> list[dict[str, Any]]:
    """Free. Individual failed calls, newest first — the raw messages behind the counts.

    Use after ``usage_errors`` when you need the actual wording of a failure, or to find a
    specific one: ``search="context length"`` matches a substring of the provider's message.
    Set ``replayable_only=True`` to keep only failures whose request body was captured.

    Args:
        since: Duration or ISO timestamp, as in ``usage_errors``.
        keys: API key **names** to restrict to. Omit for all keys.
        statuses: OutputStatus values to keep.
        operations: ``operation`` tags to keep.
        search: Substring the provider's error message must contain.
        replayable_only: Keep only failures that can be replayed via ``get_request_payload``.
        limit: Maximum rows (max 200).

    Returns:
        One row per failed call: timestamp, status, the provider's message, model, key,
        deployment, timings, tokens, and whether its body was captured. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            samples = await client.usage_error_samples(
                since=since,
                keys=keys,
                statuses=statuses,
                operations=operations,
                search=search,
                replayable_only=replayable_only,
                limit=limit,
            )
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [s.model_dump(mode="json") for s in samples]


@mcp.tool()
async def get_request_payload(log_id: str) -> dict[str, Any]:
    """Free. The request body behind a failed call, ready to send again.

    This is what turns "it failed" into "here is the exact request that failed": POST the
    returned ``payload`` to the returned ``endpoint`` to reproduce it. Get a ``log_id`` from
    ``usage_errors`` (``sample_log_id``) or ``usage_error_samples`` (``id``).

    Bodies are stored only for failures that actually reached a provider — a routing refusal
    has nothing to replay, and successes are not captured at all. Base64 attachments were
    replaced by size markers when the row was written, so a payload with images replays as
    text unless you re-attach them.

    Note this returns the prompt as it was sent, including any tool schemas.

    Args:
        log_id: The usage log id.

    Returns:
        ``payload`` (the body), ``endpoint`` (where to POST it), plus the status, the
        provider's error message and the model. Needs a **dev** API key.
    """
    try:
        async with _client() as client:
            return (await client.usage_payload(log_id)).model_dump(mode="json")
    except LLMError as exc:
        return {"error": _dev_error(exc)}


def _short_ts(value: datetime) -> str:
    """``2026-07-28T09:18:50+00:00`` → ``07-28 09:18`` (UTC; windows are ≤90 days, the year is noise)."""
    return value.astimezone(UTC).strftime("%m-%d %H:%M")


def _lean(row: dict[str, Any]) -> dict[str, Any]:
    """Drop None / empty fields — in these tools' output, absent always means zero or none."""
    return {k: v for k, v in row.items() if v is not None and v not in ("", {}, [])}


@mcp.tool()
async def usage_latency(
    since: str = "7d",
    group: str = "model",
    models: list[str] | None = None,
    deployments: list[str] | None = None,
    hosting_providers: list[str] | None = None,
    statuses: list[str] | None = None,
    buckets: list[int] | None = None,
    min_calls: int = 1,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Free. TTFT/duration percentiles and decode speed per model or deployment, busiest first.

    The tool for "is X slow" and "who serves X fastest": compare models or deployments over a
    window, on successful calls by default. Pass ``buckets`` (input-token edges, e.g.
    ``[2000, 20000]``) to split each group by prompt size — a high TTFT on tiny prompts is
    upstream queueing, one that grows with size is prefill.

    Args:
        since: Duration (``7d``, ``24h``) or ISO timestamp. Max 90 days.
        group: ``model`` or ``deployment``.
        models: Model names to keep (unknown name = error, not empty result).
        deployments: Deployment names to keep.
        hosting_providers: Hosting provider ids to keep, e.g. ``["openrouter"]``.
        statuses: Statuses to aggregate; default SUCCESS only.
        buckets: Ascending input-token edges splitting each group by prompt size.
        min_calls: Hide groups with fewer calls.
        limit: Maximum rows.

    Returns:
        Rows with calls, avg_in/avg_out tokens, ttft_p50/p90/p99 and dur_p50/p90 (ms), and
        tok_s (decode tokens/second). Absent field = none. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            rows = await client.usage_latency(
                since=since,
                group=group,
                models=models,
                deployments=deployments,
                hosting_providers=hosting_providers,
                statuses=statuses,
                buckets=buckets,
                min_calls=min_calls,
                limit=limit,
            )
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {
                "model": r.model,
                "deployment": r.deployment,
                "bucket": r.bucket,
                "calls": r.calls,
                "avg_in": r.avg_input,
                "avg_out": r.avg_output,
                "ttft_p50": r.ttft_p50,
                "ttft_p90": r.ttft_p90,
                "ttft_p99": r.ttft_p99,
                "dur_p50": r.dur_p50,
                "dur_p90": r.dur_p90,
                "tok_s": r.decode_tps,
            }
        )
        for r in rows
    ]


@mcp.tool()
async def usage_timeseries(
    since: str = "24h",
    interval: str = "1h",
    models: list[str] | None = None,
    deployments: list[str] | None = None,
    hosting_providers: list[str] | None = None,
    keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Free. Calls, failures, median TTFT and cost per time bucket — when did it change.

    The tool for "did it start at 09:15" and "is it getting worse": one row per ``interval``
    bucket over the window. Client cancellations are not counted as failures.

    Args:
        since: Duration or ISO timestamp. Max 90 days.
        interval: Bucket width — ``5m``, ``1h``, ``1d``. Window is capped at 500 buckets.
        models: Model names to keep.
        deployments: Deployment names to keep.
        hosting_providers: Hosting provider ids to keep.
        keys: API key **names** to keep (see ``list_api_keys``).

    Returns:
        Rows ``{t: "MM-DD HH:MM" (UTC), calls, fail, ttft_p50, in_tok, out_tok, cost}``.
        Absent field = zero. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            points = await client.usage_timeseries(
                since=since,
                interval=interval,
                models=models,
                deployments=deployments,
                hosting_providers=hosting_providers,
                keys=keys,
            )
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {
                "t": _short_ts(p.t),
                "calls": p.calls,
                "fail": p.failures or None,
                "ttft_p50": p.ttft_p50,
                "in_tok": p.input_tokens or None,
                "out_tok": p.output_tokens or None,
                "cost": round(p.cost, 4) or None,
            }
        )
        for p in points
    ]


@mcp.tool()
async def usage_stats(
    since: str = "24h",
    group: str = "model",
    models: list[str] | None = None,
    keys: list[str] | None = None,
    operations: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Free. Volume, error rate and spend per model, key or operation — busiest first.

    The denominators ``usage_errors`` lacks: "22 failures" only matters next to "of how many
    calls". ``err_rate`` excludes client cancellations; ``by_status`` shows every non-success
    status including those.

    Args:
        since: Duration or ISO timestamp. Max 90 days.
        group: ``model``, ``key``, ``operation``, or ``model+operation`` to compare models on the same task.
        models: Model names to keep.
        keys: API key **names** to keep.
        operations: ``operation`` tags to keep.
        limit: Maximum rows.

    Returns:
        Rows ``{name, calls, fail, err_rate, cost, in_tok, out_tok, reason_tok, by_status}``.
        Absent field = zero. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            rows = await client.usage_stats(since=since, group=group, models=models, keys=keys, operations=operations, limit=limit)
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {
                "name": r.name,
                "calls": r.calls,
                "fail": r.failures or None,
                "err_rate": r.error_rate or None,
                "cost": round(r.cost, 4) or None,
                "in_tok": r.input_tokens or None,
                "out_tok": r.output_tokens or None,
                "reason_tok": r.reasoning_tokens or None,
                "by_status": r.by_status or None,
            }
        )
        for r in rows
    ]


@mcp.tool()
async def usage_redirects(since: str = "24h", limit: int = 20) -> dict[str, Any]:
    """Free. Calls served on a model other than the one requested — the way to catch a fallback costing more than its primary.

    Args:
        since: Duration or ISO timestamp. Long windows can time out; start at ``8h``.
        limit: Maximum requested -> served pairs.

    Returns:
        ``{total, redirected, share, by_kind, pairs}``; each pair ``{kind, requested, served, calls, cost}``
        where ``kind`` is ``alias`` or ``fallback``. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            report = await client.usage_redirects(since=since, limit=limit)
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    return _lean(
        {
            "total": report.total,
            "redirected": report.redirected,
            "share": report.share or None,
            "by_kind": report.by_kind or None,
            "pairs": [
                _lean({"kind": p.kind, "requested": p.requested, "served": p.served, "calls": p.calls, "cost": round(p.cost, 2) or None})
                for p in report.pairs
            ],
        }
    )


@mcp.tool()
async def fallback_health(purpose: str = "chat") -> list[dict[str, Any]]:
    """Free. Models whose fallback chain cannot catch them — no chain, a dangling id, or no rung reachable on a plan they serve.

    A rung only counts on a plan whose hosting providers admit it, so an openrouter-only rung is
    invisible to a plan without openrouter. Empty result = every chain is sound.

    Args:
        purpose: Model purpose to audit.

    Returns:
        Rows ``{model, chain, problems, uncovered_plans}``, worst first. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            rows = await client.fallback_health(purpose=purpose)
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {"model": r.model, "chain": " > ".join(r.chain) or None, "problems": r.problems, "uncovered_plans": r.uncovered_plans or None}
        )
        for r in rows
    ]


@mcp.tool()
async def list_deployments(
    models: list[str] | None = None,
    hosting_providers: list[str] | None = None,
    statuses: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Free. Deployment configuration + health, by name — what the catalog routes to and how.

    Read-only and secret-free: shows ``provider_model_id`` (what actually goes upstream),
    ``extra_body`` (per-deployment request overrides, e.g. an OpenRouter sub-provider pin),
    price/output-token overrides, and health (status, last_error, since when — shown for
    non-ACTIVE deployments).

    Args:
        models: Model names to keep.
        hosting_providers: Hosting provider ids to keep, e.g. ``["openrouter"]``.
        statuses: Deployment statuses to keep, e.g. ``["ERROR"]``.

    Returns:
        One row per deployment, sorted by model then priority. Absent field = none/inherited.
        Needs a **dev** key.
    """
    try:
        async with _client() as client:
            rows = await client.list_deployments(models=models, hosting_providers=hosting_providers, statuses=statuses)
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {
                "deployment": r.deployment,
                "model": r.model,
                "hosting": r.hosting_provider,
                "status": r.status,
                "priority": r.priority,
                "provider_model_id": r.provider_model_id,
                "region": r.region or None,
                "extra_body": r.extra_body or None,
                "max_out": r.max_output_tokens,
                "price_in": r.input_token_price,
                "price_out": r.output_token_price,
                "price_cache": r.input_cache_price,
                "last_error": r.last_error,
                "since": _short_ts(r.status_since) if r.status_since and r.status != "ACTIVE" else None,
            }
        )
        for r in rows
    ]


@mcp.tool()
async def usage_samples(
    since: str = "24h",
    keys: list[str] | None = None,
    models: list[str] | None = None,
    deployments: list[str] | None = None,
    statuses: list[str] | None = None,
    operations: list[str] | None = None,
    search: str | None = None,
    min_ttft_ms: int | None = None,
    min_duration_ms: int | None = None,
    limit: int = 20,
    *,
    include_route: bool = False,
    include_preview: bool = False,
    replayable_only: bool = False,
) -> list[dict[str, Any]]:
    """Free. Individual calls with their timings, newest first — successes included on demand.

    ``usage_error_samples`` generalized: ``statuses=["SUCCESS"], min_ttft_ms=8000`` shows the
    actual slow calls behind a bad percentile; ``include_route=True`` attaches the route trace
    (aliases, fallbacks, retries) that explains where the time went.

    Args:
        since: Duration or ISO timestamp. Max 90 days.
        keys: API key **names** to keep.
        models: Model names to keep.
        deployments: Deployment names to keep.
        statuses: Statuses to keep — ``SUCCESS`` allowed; default is all failures.
        operations: ``operation`` tags to keep.
        search: Substring the error message must contain.
        min_ttft_ms: Keep calls whose time-to-first-token was at least this.
        min_duration_ms: Keep calls that took at least this long overall.
        include_route: Attach the route trace.
        include_preview: Attach the stripped request preview.
        replayable_only: Keep only calls whose request body was captured.
        limit: Maximum rows (max 200).

    Returns:
        Rows with ``at`` (UTC), status, names, timings and tokens; ``id`` only when the call is
        replayable via ``get_request_payload``. Absent field = zero/none. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            samples = await client.usage_samples(
                since=since,
                keys=keys,
                models=models,
                deployments=deployments,
                statuses=statuses,
                operations=operations,
                search=search,
                min_ttft_ms=min_ttft_ms,
                min_duration_ms=min_duration_ms,
                include_route=include_route,
                include_preview=include_preview,
                replayable_only=replayable_only,
                limit=limit,
            )
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    return [
        _lean(
            {
                "at": _short_ts(s.created_at),
                "status": s.status,
                "detail": s.detail or None,
                "model": s.model,
                "key": s.api_key,
                "deployment": s.deployment,
                "op": s.operation or None,
                "type": s.request_type or None,
                "dur_ms": s.duration_ms or None,
                "ttft_ms": s.ttft_ms,
                "in_tok": s.input_tokens or None,
                "out_tok": s.output_tokens or None,
                "reason_tok": s.reasoning_tokens or None,
                "finish": s.finish_reason or None,
                "route": s.route,
                "preview": s.request_preview,
                "id": s.id if s.replayable else None,
            }
        )
        for s in samples
    ]


@mcp.tool()
async def list_mcp_servers() -> list[dict[str, Any]]:
    """Free. Every MCP server the gateway exposes — slug, name, type (builtin/json/handmade), enabled.

    Secret-free: credentials are never returned. This is the MCP counterpart of ``list_deployments``.

    Returns:
        Rows ``{slug, name, type, enabled, description}``, grouped by type. Needs a **dev** key.
    """
    try:
        async with _client() as client:
            return await client.list_mcp_servers()
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]


@mcp.tool()
async def mcp_usage(since: str | None = None) -> list[dict[str, Any]]:
    """Free. Per-MCP-server tool-call volume, error rate and latency — the MCP counterpart of ``usage_stats``.

    Args:
        since: ISO timestamp lower bound (e.g. ``2026-09-01T00:00:00Z``). Omit for all-time.

    Returns:
        One row per server ``{mcp_server_id/slug, calls, errors, ...}`` from get_mcp_usage_summary.
        Needs a **dev** key.
    """
    try:
        async with _client() as client:
            return await client.mcp_usage(since=since)
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]


def main() -> None:
    """Run the MCP server over stdio (entry point for ``gate-llmax agent mcp``)."""
    mcp.run()


if __name__ == "__main__":
    main()
