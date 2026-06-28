"""``cast_json(repair=True)`` / ``extract_json_from_text(repair=True)`` salvage malformed JSON."""

from __future__ import annotations

from gate_llmax.parsing import extract_json_from_text


def test_repair_handles_unescaped_control_chars() -> None:
    # A raw newline inside a string value is invalid JSON; repair escapes it.
    bad = '{"answer": "line one\nline two"}'
    assert extract_json_from_text(bad) is None  # standard extraction fails
    repaired = extract_json_from_text(bad, repair=True)
    assert repaired == {"answer": "line one\nline two"}


def test_repair_handles_bad_backslash_escape() -> None:
    # A lone backslash (invalid escape) is doubled by the repair pass.
    bad = r'{"path": "C:\Users\x"}'
    repaired = extract_json_from_text(bad, repair=True)
    assert repaired is not None
    assert repaired["path"] == r"C:\Users\x"


def test_repair_off_by_default() -> None:
    assert extract_json_from_text('{"answer": "a\nb"}') is None


def test_well_formed_json_unaffected() -> None:
    assert extract_json_from_text('{"a": 1}', repair=True) == {"a": 1}
