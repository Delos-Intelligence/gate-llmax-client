"""Call ids: the UUIDv7 the client mints for each Gate request and sends as ``X-Gate-Call-Id``."""

from __future__ import annotations

import secrets
import sys
import time
from uuid import UUID

if sys.version_info >= (3, 14):
    from uuid import uuid7

CALL_ID_HEADER = "X-Gate-Call-Id"
"""The header naming one Gate request; Gate stores it on every log row the request writes."""


def new_call_id() -> UUID:
    """A fresh call id: a UUIDv7, so its first 48 bits are the Unix time of the dispatch in milliseconds."""
    if sys.version_info >= (3, 14):
        return uuid7()
    return uuid7_fallback()


def uuid7_fallback() -> UUID:
    """An RFC 9562 UUIDv7 for Pythons without ``uuid.uuid7``: 48-bit Unix ms, version 7, variant ``10``, random rest."""
    unix_ms = time.time_ns() // 1_000_000 & 0xFFFF_FFFF_FFFF
    return UUID(int=(unix_ms << 80) | (0x7 << 76) | (secrets.randbits(12) << 64) | (0b10 << 62) | secrets.randbits(62))
