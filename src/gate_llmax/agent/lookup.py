"""Model-name lookup for the MCP tools: fuzzy search, developer grouping, short dates."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

# Fallback only: used when the gateway is too old to send ``developer_id`` with /v1/models.
DEVELOPER_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("whisper", "openai"),
    ("text-embedding", "openai"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("veo", "google"),
    ("imagen", "google"),
    ("bge", "google"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("devstral", "mistral"),
    ("pixtral", "mistral"),
    ("voxtral", "mistral"),
    ("magistral", "mistral"),
    ("ministral", "mistral"),
    ("grok", "xai"),
    ("deepseek", "deepseek"),
    ("kimi", "moonshot"),
    ("qwen", "alibaba"),
    ("nova", "amazon"),
    ("glm", "z-ai"),
    ("eleven", "elevenlabs"),
    ("azure", "microsoft"),
    ("phi", "microsoft"),
    ("command", "cohere"),
    ("sonar", "perplexity"),
)


def tokens(name: str) -> list[str]:
    """Lowercase alphanumeric words of a model name — ``claude-4.5-opus`` → ``claude, 4, 5, opus``."""
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def developer_of(name: str, declared: str | None = None) -> str:
    """The model's developer: the gateway's ``developer_id`` when it sends one, else guessed from the name."""
    if declared:
        return declared
    lowered = name.lower()
    return next((dev for prefix, dev in DEVELOPER_BY_PREFIX if lowered.startswith(prefix)), "other")


def short_date(value: datetime | None) -> str | None:
    """Format a timestamp as ``ddmmyyyy``."""
    return value.strftime("%d%m%Y") if value is not None else None


@dataclass(frozen=True)
class Match:
    """One candidate name scored against a query; lower ``kind``/``extras`` is closer."""

    name: str
    kind: int
    extras: int


def score(query_words: list[str], name: str) -> Match | None:
    """Score ``name`` against the query words, or None when it does not match at all."""
    words = tokens(name)
    whole = sum(1 for w in query_words if w in words)
    if whole == len(query_words):
        kind = 0
    elif all(w in "".join(words) for w in query_words):
        kind = 1
    else:
        return None
    return Match(name=name, kind=kind, extras=len(words) - whole)


def search(query: str, names: Iterable[str]) -> list[str]:
    """Names matching ``query``, closest first — ``opus 5`` before ``4.5 opus``, newest on a tie."""
    query_words = tokens(query)
    if not query_words:
        return []
    matches = [m for name in sorted(names, reverse=True) if (m := score(query_words, name))]
    return [m.name for m in sorted(matches, key=lambda m: (m.kind, m.extras))]


@dataclass(frozen=True)
class Pick:
    """Outcome of a lookup: a single ``name``, else ``candidates`` — ambiguous when several matched."""

    name: str | None
    candidates: list[str]
    ambiguous: bool = False


def pick(query: str, names: Iterable[str], limit: int = 3) -> Pick:
    """Resolve ``query`` to one name, or to the ``limit`` closest when several are equally close."""
    pool = list(names)
    exact = next((n for n in pool if n.lower() == query.lower()), None)
    if exact:
        return Pick(name=exact, candidates=[exact])
    ranked = search(query, pool)
    if not ranked:
        return Pick(name=None, candidates=difflib.get_close_matches(query.lower(), pool, n=limit, cutoff=0.3))
    query_words = tokens(query)
    best = score(query_words, ranked[0])
    tied = [n for n in ranked if (m := score(query_words, n)) and best and (m.kind, m.extras) == (best.kind, best.extras)]
    if len(tied) == 1:
        return Pick(name=ranked[0], candidates=ranked[:limit])
    return Pick(name=None, candidates=ranked[:limit], ambiguous=True)
