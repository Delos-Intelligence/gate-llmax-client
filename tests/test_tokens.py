"""Token-count parity with ``llmax.tokens.count`` (no server required).

The expected values are copied verbatim from llmax's own ``test_tokencount`` and
``test_usage`` suites; ``count`` must stay byte-for-byte compatible so context
budgets tuned against llmax behave identically after migrating to Gate.
"""

from __future__ import annotations

import pytest

from gate_llmax import count

# (text, exact cl100k_base token count as pinned by llmax)
PINNED = [
    ("J'aime les légumes.", 8),
    ("Another sentence.", 4),
    (
        "To iterate over all the items in your data list and perform an assertion for each, "
        "you can use a loop within your test. This allows you to check that the count of tokens "
        "for each string in data matches the expected value.",
        47,
    ),
    ("Raconte moi une blague.", 8),
    (
        "Pourquoi les plongeurs plongent-ils toujours en arrière et jamais en avant ? Parce que sinon ils tombent dans le bateau !",
        34,
    ),
]


@pytest.mark.parametrize(("text", "expected"), PINNED)
def test_count_matches_llmax(text: str, expected: int) -> None:
    assert count(text) == expected


def test_count_special_tokens_are_safe() -> None:
    # Must not raise on reserved sequences like the end-of-text marker.
    assert count("<|endoftext|> tail") > 0
