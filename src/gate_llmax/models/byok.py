"""BYOK: a caller's own upstream provider credential, forwarded per request as a header."""

from __future__ import annotations

import base64

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
