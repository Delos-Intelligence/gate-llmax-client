"""Tests for the heavy-test suite's pure logic: case selection, judging, and profiling."""

from __future__ import annotations

from gate_llmax.agent.heavy_test import (
    CASES,
    REASONING_BASELINE_EFFORT,
    REASONING_TOKEN_FLOOR,
    Expect,
    Observed,
    case_catalogue,
    check,
    latency_block,
    profile_case,
    select_cases,
    tool_names_from_deltas,
)
from gate_llmax.models.config import ModelCapabilities, ModelInfo, ModelPurpose
from gate_llmax.types import ReasoningEffort


def model(*, tools: bool = False, images: bool = False, reasoning: bool = False, purpose: str = "chat") -> ModelInfo:
    return ModelInfo(
        id="m",
        name="m",
        purpose=ModelPurpose(purpose),
        capabilities=ModelCapabilities(supports_tools=tools, supports_images=images, supports_reasoning=reasoning),
        input_token_price=1.0,
        output_token_price=1.0,
    )


def test_text_only_model_gets_no_tool_or_vision_cases():
    picked = select_cases(model())
    assert picked
    for case in picked:
        assert not case.needs


def test_capabilities_widen_the_suite():
    plain = len(select_cases(model()))
    full = len(select_cases(model(tools=True, images=True, reasoning=True)))
    assert full == len(CASES)
    assert full > plain


def test_only_filter_accepts_ids_and_tags():
    full = model(tools=True, images=True, reasoning=True)
    by_id = select_cases(full, ["smoke"])
    assert [c.id for c in by_id] == ["smoke"]
    by_tag = select_cases(full, ["tools"])
    assert len(by_tag) > 1
    assert all("tools" in c.tags for c in by_tag)


def test_reasoning_profile_lifts_the_token_floor_and_drops_effort():
    case = next(c for c in CASES if c.id == "smoke")
    assert case.specifics.max_tokens == 16
    profiled = profile_case(case, model(reasoning=True))
    assert profiled.specifics.max_tokens == REASONING_TOKEN_FLOOR
    assert profiled.specifics.reasoning_effort is REASONING_BASELINE_EFFORT
    # `minimal` is in the enum but gpt-5.6 rejects it outright — never use it as the baseline.
    assert REASONING_BASELINE_EFFORT is not ReasoningEffort.MINIMAL


def test_reasoning_profile_leaves_the_truncation_budget_alone():
    case = next(c for c in CASES if c.id == "truncation")
    assert profile_case(case, model(reasoning=True)).specifics.max_tokens == 16


def test_reasoning_profile_leaves_the_reasoning_cases_at_full_effort():
    case = next(c for c in CASES if c.id == "reasoning")
    assert profile_case(case, model(reasoning=True)).specifics.reasoning_effort is ReasoningEffort.HIGH


def test_non_reasoning_model_is_left_untouched():
    case = next(c for c in CASES if c.id == "smoke")
    assert profile_case(case, model()) is case


def test_check_passes_a_good_answer():
    assert check(Expect(contains_all=("pong",)), Observed(text="pong")) == []


def test_check_reports_each_unmet_clause():
    failures = check(
        Expect(contains_all=("red",), forbid=("THREE",), tool_calls_min=1),
        Observed(text="THREE blue circles"),
    )
    assert any("missing 'red'" in f for f in failures)
    assert any("forbidden 'THREE'" in f for f in failures)
    assert any("tool call" in f for f in failures)


def test_reasoning_only_reply_is_distinguished_from_an_empty_one():
    reasoning_only = check(Expect(), Observed(text="", reasoning="thinking...", finish_reason="length"))
    assert any("reasoning" in f for f in reasoning_only)
    empty = check(Expect(), Observed(text=""))
    assert empty == ["empty reply"]


def test_tool_call_satisfies_non_empty():
    assert check(Expect(non_empty=True), Observed(text="", tool_names=["get_weather"])) == []


def test_json_keys_require_a_parsed_object():
    assert check(Expect(json_keys=("city",)), Observed(text="{}", json_object=None)) == ["reply did not parse as JSON"]
    assert check(Expect(json_keys=("city",)), Observed(text="{}", json_object={"country": "FR"})) == ["JSON missing keys ['city']"]
    assert check(Expect(json_keys=("city",)), Observed(text="{}", json_object={"city": "Lyon"})) == []


def test_tool_name_deltas_are_reassembled_in_index_order():
    deltas = [
        {"index": 1, "function": {"name": "get_local_time"}},
        {"index": 0, "function": {"name": "get_weather"}},
        {"index": 0, "function": {"arguments": '{"city":'}},
    ]
    assert tool_names_from_deltas(deltas) == ["get_weather", "get_local_time"]


def test_latency_block_handles_the_empty_case():
    assert latency_block([]) == {"min": 0, "p50": 0, "p95": 0, "max": 0}
    assert latency_block([10, 20, 30])["p50"] == 20


def test_multi_image_case_needs_vision_and_carries_two_images():
    case = next(c for c in CASES if c.id == "vision-multi")
    assert len(case.images) == 2
    assert case.images[0] != case.images[1]
    assert case.expect.contains_all == ("red", "blue")
    assert case not in select_cases(model())
    assert case in select_cases(model(images=True))


def test_catalogue_ids_are_unique_and_documented():
    rows = case_catalogue()
    assert len(rows) == len(CASES)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["intent"] for r in rows)
