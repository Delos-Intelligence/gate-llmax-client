---
name: gate-llmax
description: How to call LLMs through the Gate gateway with the gate-llmax Python client — the fast path for chat, streaming, structured output, tools, model fallback, and cosmos-style plans. Use when writing or editing code that sends prompts to an LLM, embeds text, or generates images/audio/video in a project that depends on gate-llmax (imports `gate_llmax` / `LLMClient`). Pairs with the `gate-llmax` MCP server for listing models/plans and building plan-covering fallback lists.
---

# Using the Gate client (gate-llmax)

One async client (`LLMClient`) over every provider. You build a request, then call a model. `operation=` is **required** on every request (it tags the usage/billing row). Everything is `async`.

## Setup

```python
from gate_llmax import LLMClient

async with LLMClient(api_key=GATE_KEY, base_url=GATE_URL) as client:
    resp = await client.request(prompt="Tell me a joke.", operation="jokes").call("gpt-4o")
    print(resp.raw_text)
```

Reuse one client (it owns a connection pool). Outside a `with`, call `await client.close()`. `api_key` is sent as `X-Gate-Key`; `base_url` is your Gate instance.

## The calls you actually make

Build with `.request(...)` (or `.simple_request(prompt, operation=..., temperature=...)` for flat tuning args), then pick a terminal:

```python
b = client.request(system_prompt="You are terse.", prompt="2+2?", operation="math")

await b.call("gpt-4o")                      # one model → LLMResponse (.raw_text, .status, .usage)
await b.call_prefer(["gpt-4o", "gpt-4o-mini"])   # try in order, first SUCCESS wins (only the winner bills)
await b.call_best(["a", "b"], greatest="quality")# fan out, return best by model extra_attributes[...]
await b.multicall(["a", "b"])               # call several, get all responses
```

Streaming:

```python
async for chunk in client.request(prompt="Hi", operation="chat").call_stream("gpt-4o", smooth=True):
    print(chunk.text, end="", flush=True)
# .stream().call_prefer([...]) streams with server-side model fallback
```

Structured output — chain `.cast(Model)` (typed) or `.cast_json()` (dict); forces JSON:

```python
from pydantic import BaseModel
class Answer(BaseModel): value: int
typed = await client.request(prompt="2+2 as JSON {value}", operation="math").cast(Answer).call("gpt-4o")
typed.value        # -> Answer(value=4)  (None if it didn't parse)
```

Tools — pass OpenAI-shaped schemas; give an `executor` to auto-run the loop:

```python
await (client.request(prompt="weather in Paris?", operation="agent")
       .with_tools(tools, executor)      # executor(id, name, args) -> result_text
       .call("gpt-4o"))
```

Other media: `client.embed(...)`, `client.image(...)`, `client.audio("speech", ...)`, `client.transcribe(...)`, `client.vision(...)`, `client.video(...)` — each returns a builder; finish with `.call(model)`.

## Routing: hosting providers & plans

A request can restrict *where* it runs:

- `.hosting("azure", "aws-bedrock")` — only these hosting providers (a canonical slug also admits its tier variants).
- `plan=` / client `default_plan=` — a **plan** is a named preset of hosting providers (a cost/infra tier, e.g. `omicron`). It resolves to that plan's providers server-side. An explicit `.hosting(...)` wins over a plan.

```python
resp = await client.request(prompt=p, operation="op").call("gpt-4o")   # plan set via default_plan on the client
```

## If the app uses plans (cosmos): cover them with call_prefer

Different users are on different plans, and **a model only resolves on plans whose hosting providers actually run it** — so a single hard-coded model can 404 for some users. The fix is one `call_prefer([...])` list ordered so that, whatever plan the user is on, the chain reaches a model available on it.

**Don't guess the list — use the `gate-llmax` MCP.** Every tool below is **free** (gateway
metadata, no model ever runs) except `heavy_test`, which is the one that spends real quota:

- `ping` — is the gateway reachable, which URL/key is configured, is it a dev key? Start here when something looks off.
- `list_plans` — the plans the gateway serves (ordered).
- `list_models` / `model_plan_matrix` — models and which plans each is reachable on.
- `plan_models(plan_id)` — models available on one plan.
- `model_in_plan(model, plan_id)` — is a specific model reachable on a plan?
- **`prefer_list(purpose, plans, prefer)`** — builds the plan-covering fallback list and returns a ready `.call_prefer([...])` snippet. Pass `prefer=[...]` for your quality order; it still guarantees coverage. Check `uncovered_plans` in the result.
- `resolve(model, plan)` — preview a model's deployments under a plan (no call made).

Typical flow: call `prefer_list` with the app's `purpose` (and `prefer` models if you have a quality order), paste its `models` into `.call_prefer(...)`. Re-run it when models/plans change — never hard-code a stale list.

The plan tools need a **dev** API key (a key with the `dev` flag). `list_models` / `resolve` work with any key.

## Actually exercising a model: heavy_test (this one costs)

`heavy_test` is the only MCP tool that calls a model, and it calls it a lot. Reach for it when you
mean to *qualify* a model — before it goes in front of users, or to prove a suspected regression.
For "is the gateway up?" use `ping`, for "does this model exist / route under this plan?" use
`resolve`, for "what would the suite run?" use `heavy_test_cases` — all free. For a cheap first
signal, the single smoke case: `heavy_test(model, n=1, only=["smoke"])`.

- `heavy_test_cases` — the catalogue (id, intent, tags, required capabilities). Free, no gateway call.
- `heavy_test(model, n=5, rate=6, only=None, plan=None)` — run the capability-matched suite `n`
  times at `rate` requests/minute. The suite is picked from the model's registered capabilities, so
  a multimodal tool-using model is tested on ~22 shapes (text, streaming, multi-turn, tool calls
  auto/forced/parallel/streamed, vision, vision+tools, image degradation, JSON mode, reasoning,
  prompt cache, determinism, `n=3`, stop sequences, truncation, long input, unicode) and a
  text-only model on the subset it can serve. Nothing is asked of a model that it cannot do.

It **spends real money and quota**: `n × len(cases)` requests. Narrow it with `only=["tools"]` (case
ids or tags) while iterating, then run the full suite once. Launches are paced, not serialized —
one every `60/rate` seconds — so slow answers overlap like real traffic.

The report gives pass rate and status breakdown, latency + TTFT percentiles, token/cost totals, a
per-case table, which deployments served the traffic, a determinism check, and every failing run
with its reason (`tool 'get_weather' never called`, `reply did not parse as JSON`,
`finish_reason='stop', expected 'length'`, `no answer text — 4490 chars of reasoning`).

On a reasoning-capable model the suite sets `reasoning_effort=minimal` and lifts small token caps
on the cases that are *not* about reasoning — otherwise the model spends the whole budget thinking
and every case degrades to "returned nothing", which tests nothing.

## Gotchas

- `operation=` is mandatory. `prompt=` and `messages=` are mutually exclusive.
- `call_prefer` bills only the winning model; `multicall`/`call_best` (client-side) bill every model that completes.
- A non-SUCCESS `LLMResponse` is returned, not raised — check `resp.status`. Auth (401), capability (422), model-not-found (404) raise typed `LLMError`s.
