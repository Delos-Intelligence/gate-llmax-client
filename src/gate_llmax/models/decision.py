"""Decision (TypeSafe System One / Jev) request and response models.

A decision call submits program ``state`` plus a set of typed ``questions`` and gets
back one typed answer per key: a Choice (pick + probabilities), a Score (continuous
level), or a Noul (binary probability). Not text, not streaming.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..types import JsonDict, JsonValue
from .request import CallControl
from .response import LLMCallRecord


class DecisionQuestion(BaseModel):
    """One typed question: ``choice`` picks from ``criteria`` (id→desc), ``score`` ranks ordered ``criteria`` levels, ``noul`` is binary."""

    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: dict[str, str] | list[str] | None = None


class DecisionRequest(CallControl):
    """Request to decide a set of questions over some program state."""

    model: str
    state: str | JsonDict | list[JsonValue]
    questions: dict[str, DecisionQuestion]


class DecisionAnswer(BaseModel):
    """One answer; which fields are set depends on the question ``type``."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "score", "noul"]
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    score: float | None = None
    distribution: dict[str, float] | list[float] | None = None
    noul: float | None = None


class DecisionResponse(LLMCallRecord):
    """Response from ``/v1/decisions`` — Gate call metadata plus one answer per question key."""

    answers: dict[str, DecisionAnswer] = Field(default_factory=dict)
