"""Client-side token counting and estimation.

Two tiers:
  - ``count``: exact ``tiktoken`` (cl100k_base) count, for callers that need to
    budget context before sending. Kept byte-for-byte compatible with
    ``llmax.tokens.count`` so budgets calibrated against llmax stay identical.
  - ``estimate_input_tokens``: a dependency-free heuristic used when a request
    times out or the connection drops and the server never reports real counts,
    so callers still get a ``RawUsage`` with ``estimated=True``.

Heuristic tier:
  - Text: ~4 characters per token (standard rule-of-thumb across most models).
  - Images: ~1 000 tokens per image (conservative GPT-4V estimate; actual cost
    depends on resolution and detail level, which the client does not know).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from gate_llmax.models.messages import Message, TextMessage
from gate_llmax.types import JsonDict

if TYPE_CHECKING:
    from tiktoken import Encoding

_CHARS_PER_TOKEN = 4
_TOKENS_PER_IMAGE = 1000
_ENCODING_NAME = "cl100k_base"


@functools.lru_cache(maxsize=1)
def _encoding() -> Encoding:
    """Load and cache the cl100k_base encoding (lazy so ``tiktoken`` stays optional at import)."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - install-time error
        msg = "tiktoken is not installed; install `tiktoken` to use gate.tokens.count()."
        raise ImportError(msg) from exc
    return tiktoken.get_encoding(_ENCODING_NAME)


def count(text: str) -> int:
    """Return the cl100k_base token count of ``text``, matching ``llmax.tokens.count``.

    Like llmax, the string is wrapped in ``repr()`` before encoding — so the quoting
    and escaping overhead is included in the count. This is intentional: it keeps
    counts identical to llmax for callers whose context budgets are tuned to it.
    Special tokens are treated as plain text (never raises on ``<|endoftext|>`` etc.).
    """
    return len(_encoding().encode(repr(text), disallowed_special=()))


def estimate_input_tokens(
    system_prompt: str | list[JsonDict],
    messages: list[Message],
    images: list[str],
) -> int:
    """Estimate the number of input tokens from the request inputs."""
    text = system_prompt if isinstance(system_prompt, str) else " ".join(str(b.get("text", "")) for b in system_prompt)
    for msg in messages:
        for block in msg.content:
            if isinstance(block, TextMessage):
                text += block.text
    return len(text) // _CHARS_PER_TOKEN + len(images) * _TOKENS_PER_IMAGE
