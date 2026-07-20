"""Tests for the plan-covering call_prefer builder."""

from __future__ import annotations

from gate_llmax.agent.prefer import build_prefer_list
from gate_llmax.models.config import ModelPlanRow, ModelPurpose


def _row(name: str, plans: list[str], purpose: str = "chat") -> ModelPlanRow:
    return ModelPlanRow(model_id=name, model_name=name, purpose=ModelPurpose(purpose), available_plan_ids=plans)


MATRIX = [
    _row("premium-a", ["gold"]),
    _row("premium-b", ["gold", "silver"]),
    _row("cheap", ["gold", "silver", "bronze"]),
    _row("embed-x", ["gold", "bronze"], purpose="embed"),
]


def test_covers_every_plan():
    result = build_prefer_list(MATRIX, purpose="chat", plans=["gold", "silver", "bronze"])
    assert not result.uncovered_plans
    for plan in ("gold", "silver", "bronze"):
        assert result.coverage[plan] is not None
        assert plan in next(r.available_plan_ids for r in MATRIX if r.model_name == result.coverage[plan])


def test_prefer_seed_comes_first_and_coverage_still_holds():
    result = build_prefer_list(MATRIX, purpose="chat", plans=["gold", "silver", "bronze"], prefer=["premium-a"])
    assert result.models[0] == "premium-a"
    # premium-a only covers gold, so a broad model must follow to cover silver/bronze.
    assert "cheap" in result.models
    assert not result.uncovered_plans


def test_purpose_filter_excludes_other_purposes():
    result = build_prefer_list(MATRIX, purpose="chat", plans=["gold", "silver", "bronze"])
    assert "embed-x" not in result.models


def test_uncovered_plan_is_reported():
    result = build_prefer_list(MATRIX, purpose="chat", plans=["gold", "platinum"])
    assert result.uncovered_plans == ["platinum"]
    assert result.coverage["platinum"] is None


def test_broadest_model_preferred_for_a_single_catch_all():
    # With no prefer seed, one broad model should cover all three plans.
    result = build_prefer_list(MATRIX, purpose="chat", plans=["bronze", "silver", "gold"])
    assert result.models == ["cheap"]


def test_snippet_is_pasteable():
    result = build_prefer_list(MATRIX, purpose="chat", plans=["gold"])
    snippet = result.snippet(operation="demo")
    assert ".call_prefer([" in snippet
    assert 'operation="demo"' in snippet
