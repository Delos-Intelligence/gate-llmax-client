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

_CONTROL_CHAR_MIN = 0x20  # below this (except \n\r\t) is an unprintable control char


def _unwrap_nested_result_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Peel ``{'result': {...}}`` layers (Hyperion ``DictWrapper`` pattern)."""
    cur: Any = data
    while isinstance(cur, dict) and list(cur.keys()) == ["result"]:
        inner = cur["result"]
        if not isinstance(inner, dict):
            break
        cur = inner
    return cur if isinstance(cur, dict) else {}


def _repair_and_load(text: str) -> dict[str, Any] | None:
    r"""Aggressive repair for malformed JSON: sanitize control chars + bad escapes, then retry.

    Replaces unprintable control chars (keeping ``\n\r\t``) with spaces and escapes raw
    newlines/tabs; failing that, doubles lone backslashes (invalid escapes). Returns the first
    candidate that parses to a dict, else ``None``.
    """
    escaped_controls = "".join(c if ord(c) >= _CONTROL_CHAR_MIN or c in "\n\r\t" else " " for c in text)
    escaped_controls = re.sub(r"(?<!\\)\n", r"\\n", escaped_controls)
    escaped_controls = re.sub(r"(?<!\\)\r", r"\\r", escaped_controls)
    escaped_controls = re.sub(r"(?<!\\)\t", r"\\t", escaped_controls)
    doubled_backslashes = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    for candidate in (escaped_controls, doubled_backslashes):
        try:
            loaded = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return _unwrap_nested_result_dict(loaded) or None
    return None


def extract_json_from_text(text: str, *, repair: bool = False) -> dict[str, Any] | None:
    """Best-effort JSON-object extraction from arbitrary assistant text.

    Strips reasoning tags, markdown fences, then tries ``json.loads`` directly; on
    failure, brace-slices the first ``{ ... }`` and retries (with a doubled-brace
    repair pass). ``repair=True`` adds an aggressive control-char / bad-escape repair
    pass when the above fail. Returns ``None`` when nothing parses to a dict.
    """
    if not text:
        return None

    cleaned = _REASONING_TAGS_RE.sub("", text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if cleaned:
        try:
            loaded = json.loads(cleaned)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return _unwrap_nested_result_dict(loaded) or None

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
            snippet = cleaned[first_brace : last_brace + 1]
            for candidate in (snippet, snippet.replace("{{", "{").replace("}}", "}")):
                try:
                    loaded = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(loaded, dict):
                    return _unwrap_nested_result_dict(loaded) or None

    return _repair_and_load(text) if repair else None


def parse_gate_text_to_model[T: BaseModel](raw_text: str, model_type: type[T]) -> T | None:
    """Parse ``LLMResponse.raw_text`` into ``model_type`` (JSON extract + ``model_validate``)."""
    parsed = extract_json_from_text(raw_text)
    if not parsed:
        return None
    try:
        return model_type.model_validate(parsed)
    except ValidationError:
        return None
