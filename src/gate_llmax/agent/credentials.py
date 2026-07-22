"""Where the Gate agent tooling keeps its gateway config.

The MCP server and the CLI both need a base URL and an API key. They come from the environment
when it is set, and otherwise from ``~/.config/gate-llmax/credentials.json`` — written by
``gate-llmax agent install``, which prompts for them, so nothing has to be exported by hand and
no secret lands in a project's ``.mcp.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CREDENTIALS_PATH = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "gate-llmax" / "credentials.json"


def load() -> tuple[str, str]:
    """Return the stored ``(base_url, api_key)``, or empty strings when there is no usable file."""
    try:
        data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    return str(data.get("base_url") or "").strip(), str(data.get("api_key") or "").strip()


def save(base_url: str, api_key: str) -> Path:
    """Write the credentials file with owner-only permissions and return its path."""
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(
        json.dumps({"base_url": base_url, "api_key": api_key}, indent=2) + "\n",
        encoding="utf-8",
    )
    CREDENTIALS_PATH.chmod(0o600)
    return CREDENTIALS_PATH


def resolve() -> tuple[str, str]:
    """Return ``(base_url, api_key)``, preferring the environment over the stored file."""
    base_url = os.environ.get("GATE_BASE_URL", "").strip()
    api_key = os.environ.get("GATE_API_KEY", "").strip()
    if base_url and api_key:
        return base_url, api_key
    stored_url, stored_key = load()
    return base_url or stored_url, api_key or stored_key
