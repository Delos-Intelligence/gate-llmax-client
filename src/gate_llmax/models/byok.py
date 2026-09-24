"""BYOK: a caller's own upstream provider credential, forwarded per request as a header."""

from __future__ import annotations

import base64
from enum import StrEnum

from pydantic import BaseModel, Field


class ByokCredential(BaseModel):
    """A caller-supplied upstream key (and optional endpoint) Gate uses in place of its pooled credential."""

    api_key: str = Field(description="The upstream provider key, used verbatim against the matched deployment.")
    endpoint: str | None = Field(default=None, description="Optional base URL / resource; subject to Gate's allowlist.")
    api_version: str | None = Field(default=None, description="Optional Azure api-version for a caller endpoint.")

    def to_header(self) -> str:
        """base64 of the JSON object Gate reads from the ``X-Gate-Upstream-Credentials`` header."""
        payload = self.model_dump_json(exclude_none=True)
        return base64.b64encode(payload.encode()).decode()


class ByokDialect(StrEnum):
    """The upstream API dialect a caller's endpoint speaks; only OpenAI-compatible is supported for now."""

    OPENAI = "openai"


class ByokPlan(BaseModel):
    """A caller's own endpoint plus a group→model table Gate serves on it, with optional fallback to managed models."""

    credential: ByokCredential = Field(description="The caller's upstream endpoint and key.")
    models: dict[str, str] = Field(default_factory=dict, description="Group name → model served on the caller's endpoint.")
    default_model: str | None = Field(default=None, description="byok model for groups absent from `models` when `cover_missing` is false.")
    dialect: ByokDialect = Field(default=ByokDialect.OPENAI, description="Upstream dialect of the caller's endpoint.")
    cover_missing: bool = Field(default=False, description="Serve groups absent from `models` with Delos-managed models.")
    cover_failure: bool = Field(default=False, description="Also fall back to Delos-managed models when the caller's endpoint fails.")

    def to_header(self) -> str:
        """base64 of the JSON object Gate reads from the ``X-Gate-Byok-Plan`` header."""
        payload = self.model_dump_json(exclude_none=True)
        return base64.b64encode(payload.encode()).decode()

    def model_for_group(self, name: str) -> str | None:
        """The byok model this plan serves for group *name* (case-insensitive), or None when the group is absent."""
        key = name.strip().lower()
        return next((model for group, model in self.models.items() if group.strip().lower() == key), None)
