"""Hammer one chat model with every request shape it can serve, then report what broke.

A model is registered with capability flags (``supports_tools``, ``supports_images``,
``supports_reasoning``). This module owns a catalogue of chat request shapes — plain text,
streaming, tool calls, vision, JSON mode, reasoning, sampling knobs, truncation, multi-turn —
each declaring what it needs. ``select_cases`` keeps the ones the model can actually serve, so a
vision model is tested on more shapes than a text-only one, and nothing 422s for being asked to
do the impossible.

``run_heavy_test`` then replays that suite ``n`` times at a fixed launch rate (requests per
minute), pacing dispatch rather than waiting on each answer, so slow requests overlap the way
real traffic does. Every run is checked against declarative expectations (did the tool call fire?
did the JSON parse? did the stop sequence hold?) and the result is a JSON-serializable report:
success rate, latency and TTFT percentiles, token and cost totals, per-case breakdown, deployment
spread, and the failures with enough context to chase them.

The catalogue mirrors the Gate dashboard sandbox presets — same prompts, same intent — so a
finding here reproduces by hand in one click.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from gate_llmax.client import LLMClient
from gate_llmax.exceptions import LLMError
from gate_llmax.models.config import ModelInfo, ModelPurpose, ResolvedDeployment
from gate_llmax.models.messages import Message
from gate_llmax.models.request import RequestSpecifics
from gate_llmax.models.response import LLMResponse, StreamChunk, ToolCall, ToolFunction
from gate_llmax.types import JsonDict, ReasoningEffort

DEFAULT_N = 5
DEFAULT_RATE_PER_MIN = 6.0
DEFAULT_OPERATION = "heavy-test"
DEFAULT_MAX_CONCURRENCY = 12
DEFAULT_BUDGET_SECONDS = 1800.0
TEXT_SAMPLE_CHARS = 240

# A reasoning model spends its output budget on thinking before it writes a word: at max_tokens=16
# it returns finish_reason=length and no text at all. So for reasoning-capable models the suite
# lifts every case's cap to this floor and asks for the cheapest effort — except on the cases whose
# whole point is the cap (`fixed_budget`) or the effort (the `reasoning` tag).
REASONING_TOKEN_FLOOR = 1024

# `low`, not `minimal`: gpt-5.6 rejects minimal outright (422 NO_VALIDATION) though the enum offers
# it, so the suite would fail every case on a technicality that has nothing to do with the case.
REASONING_BASELINE_EFFORT = ReasoningEffort.LOW

# 96x96 PNG: one red circle on a white background. Inline so vision cases need no fixture.
TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAABJElEQVR42u3cyRXDMAgFQKpx/wWlF+ecSxZbSILMfypAzM0WEKe8TSAABAgQIECAAAESQIAAAQL0"
    "ksdxfDx/B/QNyj5YUcJloVSUo5nMFEVppjFFaZoJTNFDJ88o2ugkGUUnnQyj6ESTwRQtdQYaAcoH2lNnlFE01hliFL117htFe52bRoBygGrp3DEClABUUeeyEaDR"
    "QHV1rhkBAjQTqLrOBSNAgADtAtRD51cjQIAAAQIECBAgQIAAAVoL5GMVECBAi4H8tAcEyNOzt/ntgbS/ANKCB2g9kDZgjeRGEeYAnYZZADU0GlWXkcy5QKeh3p2N"
    "MmqxWGAR0Eym1BIsN9kAKINp2rUtWNoP6FestTe05A0QIECAAAECBEgAAQIECFC1PAHHKowo1PZjPwAAAABJRU5ErkJggg=="
)

# 96x96 PNG: one blue rectangle on a white background — a second, distinct image for the multi-image case.
TEST_IMAGE_2_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAAAl0lEQVR42u3asQ0AQAgDsey/NL8BouSFT2lpXJNSWxAAAgQIECBAgAAJECBAgAABAgRIgAABAgQIEC"
    "BAAgQIECBAgAABAjQ9yN8DBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAtgN5XhAgQIAAAQIECJAAAQIECBAgQIAACRAgQIAAAQIESIAAAQIECBCgIz0u"
    "U6d0FlU1sQAAAABJRU5ErkJggg=="
)

WEATHER_TOOL: JsonDict = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

TIME_TOOL: JsonDict = {
    "type": "function",
    "function": {
        "name": "get_local_time",
        "description": "Get the current local time for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

LOG_OBSERVATION_TOOL: JsonDict = {
    "type": "function",
    "function": {
        "name": "log_observation",
        "description": "Record what is visible in an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "shape": {"type": "string"},
                "colour": {"type": "string"},
            },
            "required": ["shape", "colour"],
        },
    },
}

DISCOVER_SKILL_TOOL: JsonDict = {
    "type": "function",
    "function": {
        "name": "discover_skill",
        "description": "Load a skill, making its tools available for the rest of the conversation.",
        "parameters": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]},
    },
}

BOOK_FLIGHT_TOOL: JsonDict = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight. Only available once the travel skill has been discovered.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["origin", "destination"],
        },
    },
}


def weather_call(call_id: str = "call_weather_1", city: str = "Paris") -> ToolCall:
    """An assistant turn's request for ``get_weather`` — the history a follow-up case replays."""
    return ToolCall(id=call_id, function=ToolFunction(name="get_weather", arguments=json.dumps({"city": city, "unit": "celsius"})))


CACHE_SYSTEM_PROMPT = "You are a routing assistant for an LLM gateway. " + (
    "Follow the operator handbook: deployments are ranked by priority, then by health, then by "
    "region affinity; a rate-limited deployment is rotated away but stays eligible, an errored "
    "deployment is parked, and a decommissioned deployment is never revived automatically. " * 24
)


@dataclass(frozen=True)
class Expect:
    """What a good answer to a case looks like. Every unmet clause becomes a failure string."""

    non_empty: bool = True
    contains_all: tuple[str, ...] = ()
    contains_any: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    tool_calls_min: int = 0
    tool_names: tuple[str, ...] = ()
    json_keys: tuple[str, ...] = ()
    choices_min: int = 0
    finish_reason: str | None = None
    min_chars: int = 0
    max_chars: int | None = None


@dataclass(frozen=True)
class HeavyCase:
    """One request shape under test, plus the capabilities it needs and how it is judged."""

    id: str
    label: str
    intent: str
    tags: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    system_prompt: str = ""
    prompt: str = ""
    messages: tuple[Message, ...] = ()
    images: tuple[str, ...] = ()
    images_alternative: str | None = None
    tools: tuple[JsonDict, ...] = ()
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None
    json_mode: bool = False
    stream: bool = False
    fixed_budget: bool = False
    specifics: RequestSpecifics = field(default_factory=RequestSpecifics)
    expect: Expect = field(default_factory=Expect)


CASES: tuple[HeavyCase, ...] = (
    HeavyCase(
        id="smoke",
        label="Smoke test",
        intent="Cheapest possible round trip — proves the model answers at all.",
        tags=("baseline",),
        prompt="Reply with exactly: pong",
        specifics=RequestSpecifics(temperature=0, max_tokens=16),
        expect=Expect(contains_all=("pong",)),
    ),
    HeavyCase(
        id="long-form",
        label="Long form (TTFT + tok/s)",
        intent="Enough output to make time-to-first-token and throughput meaningful.",
        tags=("streaming", "latency"),
        system_prompt="You are a concise technical writer.",
        prompt=(
            "Explain how an LLM gateway routes a request across several deployments of the same model. Around 400 words, no bullet points."
        ),
        stream=True,
        specifics=RequestSpecifics(temperature=0.7, max_tokens=800),
        expect=Expect(min_chars=400),
    ),
    HeavyCase(
        id="multi-turn",
        label="Multi-turn context",
        intent="Third turn only answers correctly if the first two are actually in context.",
        tags=("baseline", "context"),
        messages=(
            Message.user("My favourite number is 17. Remember it."),
            Message.assistant("Noted — your favourite number is 17."),
            Message.user("Multiply my favourite number by 3. Reply with the number only."),
        ),
        specifics=RequestSpecifics(temperature=0, max_tokens=16),
        expect=Expect(contains_all=("51",)),
    ),
    HeavyCase(
        id="system-adherence",
        label="System prompt adherence",
        intent="A rigid system instruction against a prompt that invites prose.",
        tags=("baseline", "instruction"),
        system_prompt="You always answer in exactly one uppercase word. Never punctuate.",
        prompt="What is the capital of France? Feel free to elaborate on its history.",
        specifics=RequestSpecifics(temperature=0, max_tokens=32),
        expect=Expect(contains_all=("PARIS",), max_chars=20),
    ),
    HeavyCase(
        id="tool-single",
        label="Tool call — auto",
        intent="One obvious tool call; checks the model emits a well-formed function call.",
        tags=("tools",),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing.",
        prompt="What is the weather in Paris right now, in celsius?",
        tools=(WEATHER_TOOL,),
        tool_choice="auto",
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1, tool_names=("get_weather",)),
    ),
    HeavyCase(
        id="tool-required",
        label="Tool call — forced",
        intent="tool_choice=required: the model must call a tool even though the prompt is chatty.",
        tags=("tools",),
        needs=("tools",),
        prompt="Hi! Just saying hello — no need to look anything up.",
        tools=(WEATHER_TOOL,),
        tool_choice="required",
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1),
    ),
    HeavyCase(
        id="tool-parallel",
        label="Parallel tool calls",
        intent="Two tools, four answers needed — checks multi-call batching.",
        tags=("tools",),
        needs=("tools",),
        system_prompt="Gather everything you need in one turn.",
        prompt="I need the weather and the local time for both Paris and Tokyo.",
        tools=(WEATHER_TOOL, TIME_TOOL),
        tool_choice="auto",
        parallel_tool_calls=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=512),
        expect=Expect(non_empty=False, tool_calls_min=2),
    ),
    HeavyCase(
        id="tool-stream",
        label="Tool call over SSE",
        intent="Same tool call, streamed — checks tool-call delta re-assembly.",
        tags=("tools", "streaming"),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing.",
        prompt="What is the weather in Berlin in celsius?",
        tools=(WEATHER_TOOL,),
        tool_choice="required",
        stream=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1),
    ),
    HeavyCase(
        id="tool-followup",
        label="Tool result follow-up",
        intent="Assistant tool call + tool result already in history — the model must answer from it.",
        tags=("tools", "context"),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing.",
        messages=(
            Message.user("What is the weather in Paris right now, in celsius?"),
            Message.assistant_tool_calls([weather_call()]),
            Message.tool("call_weather_1", '{"city": "Paris", "temp_c": 14, "condition": "light rain"}', name="get_weather"),
        ),
        tools=(WEATHER_TOOL,),
        tool_choice="auto",
        specifics=RequestSpecifics(temperature=0, max_tokens=120),
        expect=Expect(contains_all=("14",)),
    ),
    HeavyCase(
        id="tool-discovery",
        label="Tool discovered mid-conversation",
        intent="A tool absent from turn 1 appears in the tool list afterwards; the model must use it.",
        tags=("tools", "context"),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing. Discovered tools stay available.",
        messages=(
            Message.user("I want to fly from Paris to Tokyo. Load whatever you need first."),
            Message.assistant_tool_calls(
                [ToolCall(id="call_discover_1", function=ToolFunction(name="discover_skill", arguments='{"skill": "travel"}'))],
            ),
            Message.tool("call_discover_1", '{"loaded": "travel", "new_tools": ["book_flight"]}', name="discover_skill"),
            Message.user("Great — now book it."),
        ),
        tools=(DISCOVER_SKILL_TOOL, BOOK_FLIGHT_TOOL),
        tool_choice="auto",
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1, tool_names=("book_flight",)),
    ),
    HeavyCase(
        id="tool-withdrawn",
        label="Tool withdrawn after being called",
        intent="History references a tool no longer offered — providers must not reject the request.",
        tags=("tools", "context", "degradation"),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing.",
        messages=(
            Message.user("What is the weather in Paris right now, in celsius?"),
            Message.assistant_tool_calls([weather_call()]),
            Message.tool("call_weather_1", '{"city": "Paris", "temp_c": 14, "condition": "light rain"}', name="get_weather"),
            Message.assistant("It is 14°C in Paris with light rain."),
            Message.user("Thanks. What time is it there?"),
        ),
        tools=(TIME_TOOL,),
        tool_choice="auto",
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1, tool_names=("get_local_time",)),
    ),
    HeavyCase(
        id="tool-discovery-stream",
        label="Tool discovered mid-conversation, streamed",
        intent="The discovery flow over SSE — the shape cosmos actually runs.",
        tags=("tools", "context", "streaming"),
        needs=("tools",),
        system_prompt="Use the tools you are given rather than guessing. Discovered tools stay available.",
        messages=(
            Message.user("I want to fly from Paris to Tokyo. Load whatever you need first."),
            Message.assistant_tool_calls(
                [ToolCall(id="call_discover_1", function=ToolFunction(name="discover_skill", arguments='{"skill": "travel"}'))],
            ),
            Message.tool("call_discover_1", '{"loaded": "travel", "new_tools": ["book_flight"]}', name="discover_skill"),
            Message.user("Great — now book it."),
        ),
        tools=(DISCOVER_SKILL_TOOL, BOOK_FLIGHT_TOOL),
        tool_choice="auto",
        stream=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1, tool_names=("book_flight",)),
    ),
    HeavyCase(
        id="vision",
        label="Vision — describe image",
        intent="Base64 image input; the answer should be a red circle.",
        tags=("vision",),
        needs=("images",),
        prompt="What single shape is in this image, and what colour is it? Answer in four words or fewer.",
        images=(TEST_IMAGE_B64,),
        specifics=RequestSpecifics(temperature=0, max_tokens=64),
        expect=Expect(contains_all=("red",), contains_any=("circle", "circ", "round", "disc", "dot")),
    ),
    HeavyCase(
        id="vision-multi",
        label="Vision — two images",
        intent="Two images in one turn; both must be seen, not just the first (multi-image regression).",
        tags=("vision",),
        needs=("images",),
        prompt="Two images follow. Name the colour of each, in order.",
        images=(TEST_IMAGE_B64, TEST_IMAGE_2_B64),
        specifics=RequestSpecifics(temperature=0, max_tokens=64),
        expect=Expect(contains_all=("red", "blue")),
    ),
    HeavyCase(
        id="vision-tools",
        label="Vision + tools",
        intent="Image in, structured tool call out — the two capabilities combined.",
        tags=("vision", "tools"),
        needs=("images", "tools"),
        system_prompt="Record what you see using the provided tool.",
        prompt="Look at this image and log what you see.",
        images=(TEST_IMAGE_B64,),
        tools=(LOG_OBSERVATION_TOOL,),
        tool_choice="required",
        specifics=RequestSpecifics(temperature=0, max_tokens=256),
        expect=Expect(non_empty=False, tool_calls_min=1, tool_names=("log_observation",)),
    ),
    HeavyCase(
        id="vision-degraded",
        label="Vision fallback (images_alternative)",
        intent="Image plus fallback text: a text-only model must answer from the text, not 422.",
        tags=("vision", "degradation"),
        prompt="What single shape is in this image, and what colour is it?",
        images=(TEST_IMAGE_B64,),
        images_alternative="[image unavailable: a red circle centred on a white background]",
        specifics=RequestSpecifics(temperature=0, max_tokens=64),
        expect=Expect(contains_all=("red",)),
    ),
    HeavyCase(
        id="json-mode",
        label="JSON mode",
        intent="response_format=json_object — the reply must parse as JSON with the asked-for keys.",
        tags=("structured",),
        system_prompt="Reply with JSON only.",
        # "JSON" must appear in the *user* turn: OpenAI's json_object mode requires the word in the
        # prompt, and the Responses bridge (gpt-5.6) does not count the system prompt towards it.
        prompt='Return JSON with keys "city", "country" and "population" for Lyon. Population as a number.',
        json_mode=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=200),
        expect=Expect(json_keys=("city", "country", "population")),
    ),
    HeavyCase(
        id="json-nested",
        label="JSON mode — nested schema",
        intent="A deeper object; catches models that flatten or wrap structured output.",
        tags=("structured",),
        system_prompt="Reply with JSON only.",
        prompt=(
            'Return JSON shaped {"model": {"name": string, "vendor": string}, "deployments": [{"region": string, '
            '"priority": number}]} describing two fictional deployments of a model called "atlas".'
        ),
        json_mode=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=400),
        expect=Expect(json_keys=("model", "deployments")),
    ),
    HeavyCase(
        id="reasoning",
        label="Reasoning — high effort",
        intent="A puzzle with a counter-intuitive answer; reasoning tokens should land in output cost.",
        tags=("reasoning",),
        needs=("reasoning",),
        prompt=(
            "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much "
            "does the ball cost? Answer with the number only."
        ),
        specifics=RequestSpecifics(reasoning_effort=ReasoningEffort.HIGH),
        expect=Expect(contains_any=("0.05", ".05", "5 cent", "0,05")),
    ),
    HeavyCase(
        id="reasoning-minimal",
        label="Reasoning — minimal effort",
        intent="Same model at the other end of the effort dial; checks the knob is forwarded at all.",
        tags=("reasoning",),
        needs=("reasoning",),
        prompt="What is 17 * 3? Reply with the number only.",
        specifics=RequestSpecifics(reasoning_effort=ReasoningEffort.MINIMAL, max_tokens=2000),
        expect=Expect(contains_all=("51",)),
    ),
    HeavyCase(
        id="prompt-cache",
        label="Prompt cache warm-up",
        intent="Long stable system prompt — repeats should report cached input tokens.",
        tags=("cache", "cost"),
        system_prompt=CACHE_SYSTEM_PROMPT,
        prompt="In one sentence: what happens to a rate-limited deployment?",
        specifics=RequestSpecifics(temperature=0, max_tokens=120),
        expect=Expect(),
    ),
    HeavyCase(
        id="determinism",
        label="Determinism (seed + temp 0)",
        intent="Same seed, same sampling — repeats should produce the same text.",
        tags=("sampling", "determinism"),
        prompt="Invent a name for a coffee shop. Reply with the name only.",
        specifics=RequestSpecifics(temperature=0, max_tokens=32, seed_sampling=42),
        expect=Expect(),
    ),
    HeavyCase(
        id="n-choices",
        label="n = 3 completions",
        intent="Buffered only — the extra completions come back in choices[].",
        tags=("sampling",),
        prompt="Give me one name for a coffee shop. Reply with the name only.",
        specifics=RequestSpecifics(temperature=1, max_tokens=32, n=3),
        expect=Expect(choices_min=3),
    ),
    HeavyCase(
        id="stop-sequence",
        label="Stop sequence",
        intent="Generation should halt at 'THREE' and never print it.",
        tags=("sampling",),
        prompt="Count upward in words, one per line, starting at ONE. Use capitals.",
        specifics=RequestSpecifics(temperature=0, max_tokens=200, stop=["THREE"]),
        expect=Expect(forbid=("THREE",)),
    ),
    HeavyCase(
        id="truncation",
        label="Truncation (max_tokens 16)",
        intent="Forces finish_reason=length — checks the cut-off path and its usage accounting.",
        tags=("limits", "streaming"),
        prompt="Write a detailed history of the espresso machine.",
        stream=True,
        fixed_budget=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=16),
        expect=Expect(non_empty=False, finish_reason="length"),
    ),
    HeavyCase(
        id="long-input",
        label="Long input (context + token accounting)",
        intent="A few thousand tokens of input; checks large prompts survive and are counted.",
        tags=("limits", "cost"),
        system_prompt="You answer questions about the transcript you are given.",
        prompt=(
            "Transcript:\n"
            + "Line {i}: the gateway rotated a rate-limited deployment and retried in place.\n" * 400
            + "\nHow many transcript lines mention retrying? Reply with one short sentence."
        ),
        specifics=RequestSpecifics(temperature=0, max_tokens=120),
        expect=Expect(),
    ),
    HeavyCase(
        id="unicode",
        label="Unicode round trip",
        intent="Accents, CJK and emoji through the whole pipe — catches encoding damage.",
        tags=("baseline", "encoding"),
        prompt="Repeat exactly, on one line: café 東京 🚀",
        specifics=RequestSpecifics(temperature=0, max_tokens=32),
        expect=Expect(contains_all=("café", "東京", "🚀")),
    ),
    HeavyCase(
        id="stream-unicode",
        label="Unicode over SSE",
        intent="Same round trip streamed — catches chunk-boundary mangling of multi-byte text.",
        tags=("streaming", "encoding"),
        prompt="Repeat exactly, on one line: café 東京 🚀",
        stream=True,
        specifics=RequestSpecifics(temperature=0, max_tokens=32),
        expect=Expect(contains_all=("café", "東京", "🚀")),
    ),
)

CASES_BY_ID: dict[str, HeavyCase] = {case.id: case for case in CASES}


def case_catalogue() -> list[dict[str, Any]]:
    """Every known case as plain data — id, label, intent, tags and required capabilities."""
    return [
        {
            "id": case.id,
            "label": case.label,
            "intent": case.intent,
            "tags": list(case.tags),
            "needs": list(case.needs),
            "stream": case.stream,
        }
        for case in CASES
    ]


def select_cases(model: ModelInfo, only: Sequence[str] | None = None) -> list[HeavyCase]:
    """The cases this model can serve: capability-gated, optionally narrowed to ``only`` ids/tags.

    An unsupported capability drops the case rather than failing it — asking a text-only model
    for vision proves nothing about the model, only about the request.
    """
    caps = model.capabilities
    available = {
        "tools": caps.supports_tools,
        "images": caps.supports_images,
        "reasoning": caps.supports_reasoning,
    }
    picked = [case for case in CASES if all(available.get(need, False) for need in case.needs)]
    if only:
        wanted = set(only)
        picked = [case for case in picked if case.id in wanted or wanted & set(case.tags)]
    return picked


def profile_case(case: HeavyCase, model: ModelInfo) -> HeavyCase:
    """Adapt a case to the model before it is sent.

    A reasoning model consumes its output budget thinking, so on a reasoning-capable model every
    case that is not about reasoning (or about the cap itself) gets ``reasoning_effort=minimal``
    and a token floor. Without it the whole suite degrades to "returned nothing, finish_reason
    length", which says nothing about the request shape under test.
    """
    if not model.capabilities.supports_reasoning or case.fixed_budget or "reasoning" in case.tags:
        return case
    max_tokens = case.specifics.max_tokens
    specifics = case.specifics.model_copy(
        update={
            "reasoning_effort": case.specifics.reasoning_effort or REASONING_BASELINE_EFFORT,
            "max_tokens": max(max_tokens, REASONING_TOKEN_FLOOR) if max_tokens is not None else None,
        }
    )
    return replace(case, specifics=specifics)


@dataclass
class RunRecord:
    """The outcome of one dispatched request."""

    case_id: str
    attempt: int
    ok: bool
    status: str
    failures: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    ttft_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost: float = 0.0
    api_provider: str = ""
    hosting_provider: str = ""
    region: str = ""
    deployment_id: str = ""
    pin: str = ""
    text_chars: int = 0
    reasoning_chars: int = 0
    tool_calls: list[str] = field(default_factory=list)
    text_sample: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable projection."""
        return {
            "case": self.case_id,
            "attempt": self.attempt,
            "ok": self.ok,
            "status": self.status,
            "failures": self.failures,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost": round(self.cost, 8),
            "api_provider": self.api_provider,
            "hosting_provider": self.hosting_provider,
            "region": self.region,
            "deployment_id": self.deployment_id,
            "pin": self.pin,
            "text_chars": self.text_chars,
            "reasoning_chars": self.reasoning_chars,
            "tool_calls": self.tool_calls,
            "text_sample": self.text_sample,
        }


@dataclass
class Observed:
    """What came back, normalized across the buffered and streamed paths."""

    text: str = ""
    reasoning: str = ""
    tool_names: list[str] = field(default_factory=list)
    json_object: dict[str, Any] | None = None
    choices: int = 1
    finish_reason: str | None = None
    output_tokens: int = 0


def check_text(expect: Expect, seen: Observed) -> list[str]:
    """Presence/absence clauses over the answer text (case-insensitive), plus the length bounds."""
    lowered = seen.text.lower()
    stripped = len(seen.text.strip())
    failures = [f"missing {needle!r}" for needle in expect.contains_all if needle.lower() not in lowered]
    failures += [f"forbidden {banned!r} present" for banned in expect.forbid if banned.lower() in lowered]
    if expect.contains_any and not any(n.lower() in lowered for n in expect.contains_any):
        failures.append(f"none of {list(expect.contains_any)} present")
    if expect.min_chars and len(seen.text) < expect.min_chars:
        failures.append(f"only {len(seen.text)} chars, expected >={expect.min_chars}")
    if expect.max_chars is not None and stripped > expect.max_chars:
        failures.append(f"{stripped} chars, expected <={expect.max_chars}")
    return failures


def check_shape(expect: Expect, seen: Observed) -> list[str]:
    """Clauses about the *shape* of the reply: tool calls, JSON, choices, finish reason."""
    failures: list[str] = []
    if len(seen.tool_names) < expect.tool_calls_min:
        failures.append(f"expected >={expect.tool_calls_min} tool call(s), got {len(seen.tool_names)}")
    failures += [f"tool {name!r} never called" for name in expect.tool_names if name not in seen.tool_names]
    if expect.json_keys:
        if seen.json_object is None:
            failures.append("reply did not parse as JSON")
        else:
            missing = [k for k in expect.json_keys if k not in seen.json_object]
            if missing:
                failures.append(f"JSON missing keys {missing}")
    if expect.choices_min and seen.choices < expect.choices_min:
        failures.append(f"expected >={expect.choices_min} choices, got {seen.choices}")
    if expect.finish_reason and seen.finish_reason != expect.finish_reason:
        failures.append(f"finish_reason={seen.finish_reason!r}, expected {expect.finish_reason!r}")
    return failures


def check_answered(expect: Expect, seen: Observed) -> list[str]:
    """Did the model say anything at all? Reasoning-only is called out separately from silence."""
    if not expect.non_empty or seen.text.strip() or seen.tool_names:
        return []
    # The model answered, it just never stopped thinking — a different bug from an empty reply.
    if seen.reasoning.strip():
        return [f"no answer text — {len(seen.reasoning)} chars of reasoning, finish_reason={seen.finish_reason!r}"]
    return ["empty reply"]


def check(expect: Expect, seen: Observed) -> list[str]:
    """Every expectation ``seen`` fails, as short human-readable strings."""
    return check_answered(expect, seen) + check_text(expect, seen) + check_shape(expect, seen)


def build(client: LLMClient, case: HeavyCase, *, operation: str, plan: str | None, deployment: str | None = None) -> Any:
    """A configured request builder for one case (cache off — a replay would report stale timings)."""
    builder = client.request(
        system_prompt=case.system_prompt,
        messages=list(case.messages) or None,
        prompt=case.prompt if not case.messages else None,
        images=list(case.images) or None,
        images_alternative=case.images_alternative,
        specifics=case.specifics.model_copy(),
        operation=operation,
        cache_call=False,
    )
    if plan:
        builder.plan = plan
    if deployment:
        builder = builder.dev(deployment)
    if case.tools:
        builder = builder.with_tools(list(case.tools), tool_choice=case.tool_choice, parallel_tool_calls=case.parallel_tool_calls)
    if case.json_mode:
        builder = builder.cast_json()
    return builder


def observe_response(response: LLMResponse) -> Observed:
    """Normalize a buffered response."""
    return Observed(
        text=response.raw_text or "",
        tool_names=[tc.function.name for tc in response.tool_calls or [] if tc.function],
        json_object=response.json_object,
        choices=len(response.choices) if response.choices else 1,
        output_tokens=response.usage.output_tokens,
    )


def tool_names_from_deltas(deltas: list[JsonDict]) -> list[str]:
    """Collect the function names appearing in accumulated streaming tool-call deltas."""
    names: dict[int, str] = {}
    for delta in deltas:
        index = delta.get("index")
        function = delta.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names[int(index) if isinstance(index, int) else len(names)] = name
    return [names[k] for k in sorted(names)]


async def consume_stream(builder: Any, model: str, record: RunRecord, started: float) -> Observed:
    """Drain one streamed call into ``record`` (usage, routing, TTFT) and return what was said."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    deltas: list[JsonDict] = []
    final: StreamChunk | None = None
    finish_reason: str | None = None
    first_at: float | None = None

    async for chunk in builder.stream().call(model):
        if chunk.text and first_at is None:
            first_at = time.perf_counter()
        text_parts.append(chunk.text)
        reasoning_parts.append(chunk.reasoning)
        if chunk.tool_calls_delta:
            deltas.extend(chunk.tool_calls_delta)
        # finish_reason and the usage totals arrive on different chunks — the terminal usage
        # frame carries no finish_reason, so keep the last of each rather than reading both
        # off whichever chunk happened to come last.
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        if chunk.input_tokens is not None:
            final = chunk

    record.status = "SUCCESS"
    if final is not None:
        record.input_tokens = final.input_tokens or 0
        record.output_tokens = final.output_tokens or 0
        record.cached_input_tokens = final.cached_input_tokens or 0
        record.cost = (final.input_cost or 0.0) + (final.output_cost or 0.0)
        record.api_provider = final.api_provider or ""
        record.hosting_provider = final.hosting_provider or ""
        record.region = final.region or ""
        record.ttft_ms = final.ttft_ms
    if record.ttft_ms is None and first_at is not None:
        record.ttft_ms = int((first_at - started) * 1000)

    return Observed(
        text="".join(text_parts),
        reasoning="".join(reasoning_parts),
        tool_names=tool_names_from_deltas(deltas),
        finish_reason=finish_reason,
        output_tokens=(final.output_tokens or 0) if final else 0,
    )


async def call_buffered(builder: Any, model: str, record: RunRecord) -> Observed:
    """Run one buffered call into ``record`` and return what was said."""
    response = await builder.call(model)
    record.status = response.status.value
    record.input_tokens = response.usage.input_tokens
    record.output_tokens = response.usage.output_tokens
    record.cached_input_tokens = response.usage.cached_input_tokens
    record.cost = response.usage.total_cost
    record.api_provider = response.usage.api_provider
    record.hosting_provider = response.usage.hosting_provider
    record.region = response.usage.region
    record.deployment_id = str(response.deployment_id) if response.deployment_id else ""
    return observe_response(response)


def pin_verdict(record: RunRecord, pinned: ResolvedDeployment | None) -> str:
    """How firmly this run proves the pin held: ``ok`` on an id match, ``consistent`` on host+region alone, else ``mismatch``."""
    if pinned is None:
        return ""
    if record.deployment_id:
        return "ok" if record.deployment_id == pinned.id else "mismatch"
    if not record.hosting_provider:
        return "unknown"
    same_region = not record.region or not pinned.region or record.region == pinned.region
    return "consistent" if record.hosting_provider == pinned.hosting_provider and same_region else "mismatch"


async def run_case(
    client: LLMClient,
    model: str,
    case: HeavyCase,
    attempt: int,
    *,
    operation: str,
    plan: str | None,
    deployment: str | None = None,
    pinned: ResolvedDeployment | None = None,
) -> RunRecord:
    """Dispatch one case once and judge the answer. Never raises — a failure is a record."""
    started = time.perf_counter()
    record = RunRecord(case_id=case.id, attempt=attempt, ok=False, status="ERROR")
    try:
        builder = build(client, case, operation=operation, plan=plan, deployment=deployment)
        seen = await (consume_stream(builder, model, record, started) if case.stream else call_buffered(builder, model, record))
        record.failures = check(case.expect, seen) if record.status == "SUCCESS" else ["status " + record.status]
        record.text_chars = len(seen.text)
        record.reasoning_chars = len(seen.reasoning)
        record.text_sample = seen.text[:TEXT_SAMPLE_CHARS]
        record.tool_calls = seen.tool_names
        record.ok = record.status == "SUCCESS" and not record.failures
    except LLMError as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.failures = [record.error]
    except (TimeoutError, asyncio.CancelledError) as exc:
        record.status = "TIMEOUT"
        record.error = f"{type(exc).__name__}: {exc}"
        record.failures = [record.error]
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.failures = [record.error]
    record.latency_ms = int((time.perf_counter() - started) * 1000)
    record.pin = pin_verdict(record, pinned)
    if record.pin == "mismatch":
        record.ok = False
        record.failures = [*record.failures, f"served by {record.deployment_id or record.hosting_provider}, not the pinned deployment"]
    return record


def percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile of ``values`` (0 when empty)."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def latency_block(values: list[int]) -> dict[str, int]:
    """Min / p50 / p95 / max for a list of millisecond timings."""
    return {
        "min": min(values) if values else 0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0,
    }


def tally(records: list[RunRecord]) -> tuple[dict[str, int], dict[str, int]]:
    """Status histogram and deployment (hosting provider / region) histogram."""
    statuses: dict[str, int] = {}
    hosts: dict[str, int] = {}
    for record in records:
        statuses[record.status] = statuses.get(record.status, 0) + 1
        if record.hosting_provider:
            key = f"{record.hosting_provider}/{record.region}" if record.region else record.hosting_provider
            hosts[key] = hosts.get(key, 0) + 1
    return statuses, hosts


def deployment_tally(records: list[RunRecord], names: Mapping[str, str]) -> dict[str, int]:
    """How many runs each deployment served, named where the id came back; a streamed run only knows its host."""
    counts: dict[str, int] = {}
    for record in records:
        if record.deployment_id:
            key = f"{names[record.deployment_id]} ({record.deployment_id})" if record.deployment_id in names else record.deployment_id
        elif record.hosting_provider:
            key = f"{record.hosting_provider} (streamed — no deployment id on the wire)"
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def pin_report(pinned: ResolvedDeployment, records: list[RunRecord]) -> dict[str, Any]:
    """Whether every run really went to the pinned deployment, and how strong the evidence for that is."""
    verdicts: dict[str, int] = {}
    for record in records:
        verdicts[record.pin or "unknown"] = verdicts.get(record.pin or "unknown", 0) + 1
    return {
        "id": pinned.id,
        "name": pinned.name,
        "hosting_provider": pinned.hosting_provider,
        "region": pinned.region,
        "status": pinned.status,
        "runs": verdicts,
        "held": verdicts.get("mismatch", 0) == 0,
        "evidence": "ok = the run reported this deployment id; consistent = streamed, only host and region could be checked",
    }


def per_case_table(cases: list[HeavyCase], records: list[RunRecord]) -> list[dict[str, Any]]:
    """One row per case: how it fared, how long it took, and every distinct failure it produced."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        runs = [r for r in records if r.case_id == case.id]
        if not runs:
            continue
        case_ok = [r for r in runs if r.ok]
        rows.append(
            {
                "case": case.id,
                "label": case.label,
                "runs": len(runs),
                "ok": len(case_ok),
                "pass_rate": round(len(case_ok) / len(runs), 3),
                "latency_ms": latency_block([r.latency_ms for r in runs if r.status == "SUCCESS"]),
                "avg_output_tokens": round(sum(r.output_tokens for r in runs) / len(runs), 1),
                "avg_reasoning_chars": round(sum(r.reasoning_chars for r in runs) / len(runs), 1),
                "failures": sorted({f for r in runs for f in r.failures}),
            }
        )
    return rows


def determinism_check(records: list[RunRecord]) -> dict[str, Any] | None:
    """Did the seeded, temperature-0 case give the same answer every time? ``None`` if not run twice."""
    runs = [r for r in records if "determinism" in CASES_BY_ID[r.case_id].tags and r.status == "SUCCESS"]
    if len(runs) < 2:
        return None
    distinct = {r.text_sample.strip() for r in runs}
    return {"runs": len(runs), "distinct_texts": len(distinct), "stable": len(distinct) == 1}


def summarize(
    model_name: str,
    model: ModelInfo,
    cases: list[HeavyCase],
    records: list[RunRecord],
    *,
    n: int,
    rate: float,
    elapsed_s: float,
    include_runs: bool,
    pinned: ResolvedDeployment | None = None,
    deployment_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fold the run records into the report the caller (or an agent) reads."""
    ok = [r for r in records if r.ok]
    latencies = [r.latency_ms for r in records if r.status == "SUCCESS"]
    ttfts = [r.ttft_ms for r in records if r.ttft_ms is not None]
    statuses, hosts = tally(records)
    per_case = per_case_table(cases, records)
    determinism = determinism_check(records)
    cached = sum(r.cached_input_tokens for r in records)
    report: dict[str, Any] = {
        "model": model_name,
        "purpose": model.purpose.value,
        "capabilities": model.capabilities.model_dump(),
        "plan_cases_selected": [c.id for c in cases],
        "requested": {"n": n, "rate_per_min": rate, "total_requests": len(records)},
        "duration_s": round(elapsed_s, 1),
        "achieved_rate_per_min": round(len(records) / (elapsed_s / 60), 2) if elapsed_s > 0 else 0.0,
        "totals": {
            "requests": len(records),
            "passed": len(ok),
            "failed": len(records) - len(ok),
            "pass_rate": round(len(ok) / len(records), 3) if records else 0.0,
            "statuses": statuses,
        },
        "latency_ms": latency_block(latencies),
        "ttft_ms": latency_block([t for t in ttfts if t is not None]),
        "tokens": {
            "input": sum(r.input_tokens for r in records),
            "output": sum(r.output_tokens for r in records),
            "cached_input": cached,
        },
        "cost_usd": round(sum(r.cost for r in records), 6),
        "hosting_providers": hosts,
        "deployments": deployment_tally(records, deployment_names or {}),
        "per_case": per_case,
        "determinism": determinism,
        "failures": [r.as_dict() for r in records if not r.ok][:60],
    }
    if pinned is not None:
        report["deployment_pin"] = pin_report(pinned, records)
    if include_runs:
        report["runs"] = [r.as_dict() for r in records]
    return report


async def deployment_index(client: LLMClient, model_name: str) -> dict[str, ResolvedDeployment]:
    """Every deployment of ``model_name`` by id, whatever its status; empty when the gateway will not say."""
    try:
        resolved = await client.resolve(model_name)
    except LLMError:
        return {}
    return {d.id: d for d in resolved.all_deployments}


async def dev_key_error(client: LLMClient) -> str | None:
    """The message to report when the configured key lacks the ``dev`` flag a deployment pin needs."""
    try:
        await client.list_plans()
    except LLMError as exc:
        return f"Pinning a deployment needs a dev API key (the gateway 403s otherwise); this one was refused: {exc}"
    return None


async def run_heavy_test(
    client: LLMClient,
    model_name: str,
    *,
    n: int = DEFAULT_N,
    rate: float = DEFAULT_RATE_PER_MIN,
    plan: str | None = None,
    deployment: str | None = None,
    only: Sequence[str] | None = None,
    operation: str = DEFAULT_OPERATION,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    include_runs: bool = False,
) -> dict[str, Any]:
    """Run the capability-matched suite ``n`` times against ``model_name`` at ``rate`` requests/min.

    Dispatch is paced, not serialized: a request is launched every ``60 / rate`` seconds whatever
    the previous one is doing, so a slow model overlaps requests exactly as production traffic
    would. ``max_concurrency`` caps how many may be in flight at once, and ``budget_seconds`` stops
    *launching* new requests once the wall-clock budget is spent (those already in flight finish).

    Args:
        client: a configured ``LLMClient`` pointed at the gateway.
        model_name: the chat model to hammer, as registered on the gateway.
        n: how many times to replay the whole suite (total requests = n x selected cases).
        rate: launch rate in requests per minute (6 = one every 10 s).
        plan: optional hosting plan to route under; ignored when ``deployment`` pins the route.
        deployment: pin every request to this deployment id (dev key only) — INACTIVE rows included, no fallback.
        only: restrict to these case ids or tags (e.g. ``["tools", "streaming"]``).
        operation: usage tag written to the gateway's usage log.
        max_concurrency: ceiling on in-flight requests.
        budget_seconds: stop launching once this much wall-clock has passed.
        include_runs: include every individual run record in the report.

    Returns:
        A JSON-serializable report: pass rate, latency/TTFT percentiles, token and cost totals,
        per-case breakdown, deployment spread and the failing runs.
    """
    models = await client.list_models()
    model = next((m for m in models if m.name.lower() == model_name.lower()), None)
    if model is None:
        return {"error": f"Model '{model_name}' is not registered on this gateway."}
    if model.purpose is not ModelPurpose.CHAT:
        return {"error": f"Model '{model_name}' has purpose '{model.purpose.value}'; heavy_test only runs chat models."}

    catalogue = await deployment_index(client, model.name)
    pinned: ResolvedDeployment | None = None
    if deployment:
        pinned = catalogue.get(deployment)
        if pinned is None:
            known = ", ".join(f"{d.name}={i}" for i, d in catalogue.items())
            return {"error": f"Deployment '{deployment}' does not serve model '{model.name}'. Its deployments: {known or 'none reported'}."}
        refused = await dev_key_error(client)
        if refused:
            return {"error": refused}

    cases = [profile_case(case, model) for case in select_cases(model, only)]
    if not cases:
        return {"error": f"No test cases match model '{model_name}' with only={list(only or [])}."}

    schedule = [(case, attempt) for attempt in range(1, n + 1) for case in cases]
    interval = 60.0 / rate if rate > 0 else 0.0
    limiter = asyncio.Semaphore(max_concurrency)

    async def dispatch(case: HeavyCase, attempt: int) -> RunRecord:
        async with limiter:
            return await run_case(client, model.name, case, attempt, operation=operation, plan=plan, deployment=deployment, pinned=pinned)

    started = time.perf_counter()
    tasks: list[asyncio.Task[RunRecord]] = []
    for index, (case, attempt) in enumerate(schedule):
        if index and interval:
            await asyncio.sleep(interval)
        if time.perf_counter() - started > budget_seconds:
            break
        tasks.append(asyncio.create_task(dispatch(case, attempt)))
    records = list(await asyncio.gather(*tasks))
    elapsed = time.perf_counter() - started

    return summarize(
        model.name,
        model,
        cases,
        records,
        n=n,
        rate=rate,
        elapsed_s=elapsed,
        include_runs=include_runs,
        pinned=pinned,
        deployment_names={i: d.name for i, d in catalogue.items()},
    )
