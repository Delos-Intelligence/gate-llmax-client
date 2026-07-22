"""Gate MCP server — the minimum an agent needs to use Gate correctly.

Exposes tools over a running Gate gateway: list models, list plans, check whether a model is
reachable on a plan, build a ``call_prefer([...])`` fallback list that covers every plan an app
serves, and — the one that spends money — ``heavy_test``, which hammers a chat model with every
request shape it can serve and reports what broke.

Config comes from the environment, falling back to the file ``gate-llmax agent install`` writes:
    GATE_BASE_URL   base URL of the Gate gateway (e.g. https://gate.example.com)
    GATE_API_KEY    a Gate API key. Plan tools need a **dev** key (the `dev` flag); model/resolve
                    tools work with any key.

Run it with ``gate-llmax agent mcp`` (stdio). Requires the ``agent`` extra: ``pip install
"gate-llmax[agent]"`` / ``uv add "gate-llmax[agent]"``.
"""

from __future__ import annotations

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
async def list_models(purpose: str | None = None) -> list[dict[str, Any]]:
    """List models registered on the gateway (id, name, purpose, capabilities, prices).

    Pass ``purpose`` (chat, embed, audio, vision, images, tts, audio_isolation, dubbing, video)
    to filter. Works with any API key.
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
    """List the gateway's hosting plans (cost/infra tiers), ordered by ``sort_order``.

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
    """For every model, the plans it is reachable on (has a deployment on an admitted host).

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
    """List the models reachable on a given plan (optionally filtered by ``purpose``).

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
    """Answer: is ``model`` reachable on ``plan_id``? Returns the model's full plan set too.

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
    """Build a ``call_prefer([...])`` fallback list that covers every plan an app serves.

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
    """Preview what a chat call to ``model`` (optionally under ``plan``) would route to.

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
    """List the request shapes ``heavy_test`` can run (id, intent, tags, required capabilities).

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
    """Hammer one chat model with every request shape it can serve, then report what broke.

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


def main() -> None:
    """Run the MCP server over stdio (entry point for ``gate-llmax agent mcp``)."""
    mcp.run()


if __name__ == "__main__":
    main()
