"""Decision builder: state/questions/operation flow onto the request, and ``.call`` sets the model."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import gate_llmax.client as client_mod
from gate_llmax import DecisionAnswer, DecisionQuestion, DecisionResponse, LLMClient
from gate_llmax.models.response import RawUsage
from gate_llmax.types import OutputStatus


def test_decision_request_shape() -> None:
    client = LLMClient(api_key="k", base_url="http://x")
    questions = {"urgent": DecisionQuestion(type="noul", instructions="Is the message urgent?")}
    builder = client.decision("help me now", questions=questions, operation="triage.urgency", timeout=30)
    assert builder.request.state == "help me now"
    assert builder.request.questions["urgent"].type == "noul"
    assert builder.request.operation == "triage.urgency"
    assert builder.request.timeout == 30
    assert builder.path == "/v1/decisions"


def test_decision_call_sets_model(monkeypatch: Any) -> None:
    async def _post_json(self: LLMClient, path: str, request: Any, label: str, *, client_timeout: float | None = None) -> Any:  # noqa: ARG001
        return type("R", (), {"json": lambda _self: {"model": request.model, "answers": {"urgent": {"type": "noul", "noul": 0.99}}}})()

    monkeypatch.setattr(client_mod.LLMClient, "_post_json", _post_json)
    client = LLMClient(api_key="k", base_url="http://x")
    questions = {"urgent": DecisionQuestion(type="noul", instructions="Is it urgent?")}

    resp = asyncio.run(client.decision("now", questions=questions, operation="t").call("jev-latest"))
    assert isinstance(resp, DecisionResponse)
    assert resp.model == "jev-latest"
    assert resp.answers["urgent"].noul == 0.99


def test_decision_answer_choice_fields() -> None:
    resp = DecisionResponse(
        model="jev-1.13.0",
        status=OutputStatus.SUCCESS,
        usage=RawUsage(model="jev-1.13.0"),
        answers={"category": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "other": 0.2}, "confidence": 0.9}},
    )
    assert resp.answers["category"].choice == "billing"
    assert resp.answers["category"].probabilities == {"billing": 0.8, "other": 0.2}


def test_decision_question_builders() -> None:
    c = DecisionQuestion.choice("Which folder?", {"work": "job", "promo": "marketing"})
    assert c.type == "choice"
    assert c.criteria == {"work": "job", "promo": "marketing"}

    s = DecisionQuestion.score("How complex?", ["trivial", "simple", "hard"])
    assert s.type == "score"
    assert s.criteria == ["trivial", "simple", "hard"]

    n = DecisionQuestion.noul("Is it spam?")
    assert n.type == "noul"
    assert n.criteria is None


def test_choice_rejects_list_criteria() -> None:
    # A bare list is rejected server-side as NO_VALIDATION; fail early and clearly instead.
    with pytest.raises(ValueError, match="dict of id"):
        DecisionQuestion(type="choice", instructions="pick", criteria=["a", "b"])


def test_decision_answer_is_yes() -> None:
    assert DecisionAnswer(type="noul", noul=0.9).is_yes()
    assert not DecisionAnswer(type="noul", noul=0.2).is_yes()
    assert DecisionAnswer(type="noul", noul=0.6).is_yes(threshold=0.5)
    assert not DecisionAnswer(type="noul", noul=0.6).is_yes(threshold=0.7)
    assert not DecisionAnswer(type="noul", noul=None).is_yes()
