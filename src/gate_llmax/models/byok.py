"""BYOK: a caller's own upstream endpoint, key and models, forwarded per request as one plan header."""

from __future__ import annotations

import base64
from enum import StrEnum

from pydantic import BaseModel, Field


def _norm_group(name: str) -> str:
    """Group name normalized for matching: trimmed, lower-cased, optional ``group_`` prefix dropped."""
    return name.strip().lower().removeprefix("group_")


class ByokCredential(BaseModel):
    """A caller-supplied upstream endpoint and key Gate uses in place of its pooled credential."""

    api_key: str = Field(description="The upstream provider key, used verbatim against the caller's endpoint.")
    endpoint: str | None = Field(default=None, description="Base URL / resource of the caller's endpoint; subject to Gate's allowlist.")
    api_version: str | None = Field(default=None, description="Optional Azure api-version for a caller endpoint.")


class ByokDialect(StrEnum):
    """The upstream API dialect a caller's endpoint speaks; only OpenAI-compatible is supported for now."""

    OPENAI = "openai"


class ByokPlan(BaseModel):
    """A caller's endpoint plus the models on it and an optional group→model table, the single BYOK request shape."""

    credential: ByokCredential = Field(description="The caller's upstream endpoint and key.")
    models: list[str] = Field(default_factory=list, description="The caller's model names on their endpoint, each callable verbatim.")
    groups: dict[str, str] = Field(default_factory=dict, description="Group name → a model in `models`, routing a group slug to it.")
    default_model: str | None = Field(default=None, description="byok model for names absent from `models`/`groups` when not covered.")
    dialect: ByokDialect = Field(default=ByokDialect.OPENAI, description="Upstream dialect of the caller's endpoint.")
    cover_missing: bool = Field(default=False, description="Serve names absent from `models`/`groups` with Delos-managed models.")
    cover_failure: bool = Field(default=False, description="Also fall back to Delos-managed models when the caller's endpoint fails.")

    def to_header(self) -> str:
        """base64 of the JSON object Gate reads from the ``X-Gate-Byok-Plan`` header."""
        payload = self.model_dump_json(exclude_none=True)
        return base64.b64encode(payload.encode()).decode()

    def model_for_name(self, name: str) -> str | None:
        """The caller's model matching *name* verbatim (case-insensitive), in its stored casing; else None."""
        key = name.strip().lower()
        return next((model for model in self.models if model.strip().lower() == key), None)

    def model_for_group(self, name: str) -> str | None:
        """The caller's model this plan attributes to group *name* (``group_`` optional, case-insensitive); else None."""
        key = _norm_group(name)
        return next((model for group, model in self.groups.items() if _norm_group(group) == key), None)
