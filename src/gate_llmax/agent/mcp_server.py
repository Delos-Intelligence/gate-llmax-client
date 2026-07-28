"""Gate MCP server — the minimum an agent needs to use Gate correctly.

Two kinds of tools, and the difference matters:

* **Free** — gateway metadata, no model ever runs, safe to call as often as you like:
  ``ping``, ``list_models``, ``list_plans``, ``model_plan_matrix``, ``plan_models``,
  ``model_in_plan``, ``prefer_list``, ``resolve``, ``heavy_test_cases``, ``verify_probes``,
  ``list_api_keys``, ``usage_errors``, ``usage_error_samples``, ``get_request_payload``.
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
from typing import Any

from gate_llmax.agent import credentials
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
async def list_models(purpose: str | None = None) -> list[dict[str, Any]]:
    """Free. List models registered on the gateway (id, name, purpose, capabilities, prices).

    Pass ``purpose`` (chat, embed, audio, vision, images, tts, audio_isolation, dubbing, video)
    to filter. Metadata only — no model is called. Works with any API key.
    """
    try:
        async with _client() as client:
            models = await client.list_models()
    except LLMError as exc:
        return [{"error": _dev_error(exc)}]
    rows = [m.model_dump(mode="json") for m in models]
    if purpose:
        rows = [r for r in rows if r.get("purpose") == purpose]
    return rows


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
    out = [r.model_dump(mode="json") for r in rows]
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

    Needs a **dev** API key. Match is on the model's registered name (case-insensitive).
    """
    try:
        async with _client() as client:
            rows = await client.model_plan_matrix()
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    row = next((r for r in rows if r.model_name.lower() == model.lower()), None)
    if row is None:
        return {"model": model, "plan_id": plan_id, "found": False, "available": False, "available_plan_ids": []}
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


@mcp.tool()
async def resolve(model: str, plan: str | None = None) -> dict[str, Any]:
    """Free. Preview what a chat call to ``model`` (optionally under ``plan``) would route to.

    Returns the resolved model and its candidate deployments without making a call — use it to
    confirm a model actually has a deployment under a plan. Works with any API key.
    """
    try:
        async with _client() as client:
            resolved = await client.resolve(model, plan=plan)
    except LLMError as exc:
        return {"error": _dev_error(exc)}
    return resolved.model_dump(mode="json")


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
        plan: optional hosting plan to route under.
        only: restrict to these case ids or tags (e.g. ``["tools"]``, ``["smoke", "streaming"]``).
        operation: usage tag written to the gateway usage log.
        max_concurrency: ceiling on in-flight requests.
        budget_seconds: stop launching new requests after this much wall-clock (in-flight ones finish).
        include_runs: return every individual run record, not just the summary and the failures.

    Returns:
        Pass rate and status breakdown, latency and TTFT percentiles, token/cost totals, a
        per-case table, which deployments served the traffic, a determinism check, and the
        failing runs with their reasons. Works with any API key.
    """
    try:
        async with _client() as client:
            return await run_heavy_test(
                client,
                model,
                n=n,
                rate=rate,
                plan=plan,
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


def main() -> None:
    """Run the MCP server over stdio (entry point for ``gate-llmax agent mcp``)."""
    mcp.run()


if __name__ == "__main__":
    main()
