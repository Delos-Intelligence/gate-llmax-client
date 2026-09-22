"""Decision (TypeSafe System One / Jev) request and response models.

A decision call submits program ``state`` plus a set of typed ``questions`` and gets
back one typed answer per key: a Choice (pick + probabilities), a Score (continuous
level), or a Noul (binary probability). Not text, not streaming.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..types import JsonDict, JsonValue
from .request import CallControl
from .response import LLMCallRecord


class DecisionQuestion(BaseModel):
    """One typed question: ``choice`` picks from ``criteria`` (id→desc), ``score`` ranks ordered ``criteria`` levels, ``noul`` is binary."""

    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> DecisionQuestion:
        # A ``choice`` needs id→description; a bare list is rejected server-side as
        # NO_VALIDATION, so catch it here with a message that says what to change.
        if self.type == "choice" and not isinstance(self.criteria, dict):
            msg = "choice questions need `criteria` as a dict of id -> description"
            raise ValueError(msg)
        return self

    @classmethod
    def choice(cls, instructions: str, criteria: dict[str, str]) -> DecisionQuestion:
        """Pick one label from ``criteria`` (id -> description)."""
        return cls(type="choice", instructions=instructions, criteria=criteria)

    @classmethod
    def score(cls, instructions: str, levels: list[str]) -> DecisionQuestion:
        """Rate on the ordered ``levels`` (low -> high); the answer carries a continuous ``score``."""
        return cls(type="score", instructions=instructions, criteria=levels)

    @classmethod
    def noul(cls, instructions: str) -> DecisionQuestion:
        """Binary yes/no; the answer's ``noul`` is the probability of yes (0..1)."""
        return cls(type="noul", instructions=instructions)


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

    def is_yes(self, threshold: float = 0.5) -> bool:
        """For a ``noul`` answer: True when the yes-probability clears ``threshold``."""
        return self.noul is not None and self.noul >= threshold


class DecisionResponse(LLMCallRecord):
    """Response from ``/v1/decisions`` — Gate call metadata plus one answer per question key."""

    answers: dict[str, DecisionAnswer] = Field(default_factory=dict)
