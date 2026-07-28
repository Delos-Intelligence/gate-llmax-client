"""Tests for the fuzzy model-name lookup behind the MCP model tools."""

from __future__ import annotations

from datetime import UTC, datetime

from gate_llmax.agent import lookup

NAMES = [
    "claude-4.5-opus",
    "claude-4.8-opus",
    "claude-5-opus",
    "claude-5-sonnet",
    "claude-opus-latest",
    "gemini-3.0-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gpt-5.6-terra",
]


def test_two_words_pin_one_model():
    assert lookup.pick("opus 5", NAMES).name == "claude-5-opus"


def test_exact_name_wins_over_fuzzy():
    assert lookup.pick("claude-4.5-opus", NAMES).name == "claude-4.5-opus"


def test_vague_query_returns_three_closest():
    found = lookup.pick("opus", NAMES)
    assert found.name is None
    assert found.ambiguous
    assert len(found.candidates) == 3
    assert "claude-5-opus" in found.candidates
    assert all("opus" in c for c in found.candidates)


def test_family_query_prefers_the_shortest_match():
    assert lookup.pick("gemini flash", NAMES).name == "gemini-3.5-flash"


def test_unknown_query_resolves_to_nothing():
    found = lookup.pick("banana", NAMES)
    assert found.name is None
    assert not found.ambiguous


def test_search_filters_and_ranks():
    assert lookup.search("sonnet", NAMES) == ["claude-5-sonnet"]
    assert lookup.search("", NAMES) == []


def test_developer_falls_back_to_the_name_prefix():
    assert lookup.developer_of("claude-5-opus") == "anthropic"
    assert lookup.developer_of("something-odd") == "other"
    assert lookup.developer_of("claude-5-opus", "acme") == "acme"


def test_short_date_is_ddmmyyyy():
    assert lookup.short_date(datetime(2026, 7, 27, 9, 30, tzinfo=UTC)) == "27072026"
    assert lookup.short_date(None) is None
