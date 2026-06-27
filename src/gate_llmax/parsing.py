"""Extract JSON from assistant ``raw_text`` and validate with Pydantic (Hyperion-style).

String bodies use the same brace-slice and markdown cleanup as Hyperion's
``JSONParser.extract_json``; nested ``{"result": {...}}`` shells are unwrapped like
``DictWrapper`` in Hyperion's processing models.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_REASONING_TAGS_RE = re.compile(
    r"<(thinking|reasoning|thought|reflection)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def _unwrap_nested_result_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Peel ``{'result': {...}}`` layers (Hyperion ``DictWrapper`` pattern)."""
    cur: Any = data
    while isinstance(cur, dict) and list(cur.keys()) == ["result"]:
        inner = cur["result"]
        if not isinstance(inner, dict):
            break
        cur = inner
    return cur if isinstance(cur, dict) else {}


def extract_json_from_text(text: str) -> dict[str, Any] | None:  # noqa: PLR0911
    """Best-effort JSON-object extraction from arbitrary assistant text.

    Strips reasoning tags, markdown fences, then tries ``json.loads`` directly; on
    failure, brace-slices the first ``{ ... }`` and retries (with a doubled-brace
    repair pass). Returns ``None`` when nothing parses to a dict.
    """
    if not text:
        return None

    cleaned = _REASONING_TAGS_RE.sub("", text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None

    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return _unwrap_nested_result_dict(loaded) or None

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return None

    snippet = cleaned[first_brace : last_brace + 1]
    try:
        loaded = json.loads(snippet)
    except json.JSONDecodeError:
        try:
            loaded = json.loads(snippet.replace("{{", "{").replace("}}", "}"))
        except json.JSONDecodeError:
            return None
    if isinstance(loaded, dict):
        return _unwrap_nested_result_dict(loaded) or None
    return None


def parse_gate_text_to_model[T: BaseModel](raw_text: str, model_type: type[T]) -> T | None:
    """Parse ``LLMResponse.raw_text`` into ``model_type`` (JSON extract + ``model_validate``)."""
    parsed = extract_json_from_text(raw_text)
    if not parsed:
        return None
    try:
        return model_type.model_validate(parsed)
    except ValidationError:
        return None
