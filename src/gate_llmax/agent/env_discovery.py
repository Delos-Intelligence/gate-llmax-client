"""Find gateway credentials already sitting in a project's env files.

Consumer projects that talk to Gate already carry a base URL and an API key in a ``.env`` file,
usually under a project prefix (cosmos uses ``COSMOS_GATE_BASE_URL`` / ``COSMOS_GATE_API_KEY``).
``gate-llmax agent install`` reads them from there instead of asking the user to paste secrets
they have already written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEARCH_DEPTH = 2
SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".next"})
# Consulted in this order: a real `.env.local` secret beats the placeholder in `.env.example`.
ENV_FILE_ORDER = (".env.local", ".env.development", ".env", ".env.example")


@dataclass(frozen=True)
class Found:
    """A credential pair discovered in an env file."""

    base_url: str
    api_key: str
    source: Path


def _env_files(project: Path) -> list[Path]:
    """Env files under ``project``, nearest first, ranked by how likely they hold real secrets."""
    files: list[Path] = []
    for path in project.rglob(".env*"):
        if not path.is_file():
            continue
        relative = path.relative_to(project)
        if len(relative.parts) > SEARCH_DEPTH or SKIP_DIRS.intersection(relative.parts):
            continue
        files.append(path)
    order = {name: index for index, name in enumerate(ENV_FILE_ORDER)}
    return sorted(files, key=lambda p: (len(p.relative_to(project).parts), order.get(p.name, len(order))))


def _parse(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` lines, ignoring comments, ``export`` prefixes and quotes."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


def discover(project: Path) -> Found | None:
    """Return the first complete ``GATE_BASE_URL`` / ``GATE_API_KEY`` pair found under ``project``.

    Keys match on suffix, so any project prefix (``COSMOS_``, ``APP_``, …) is picked up. Both
    halves must come from the same file — mixing a URL from one env file with a key from another
    is how you end up authenticating against the wrong gateway.
    """
    for path in _env_files(project):
        values = _parse(path)
        base_url = next((v for k, v in values.items() if k.endswith("GATE_BASE_URL")), "")
        api_key = next((v for k, v in values.items() if k.endswith("GATE_API_KEY")), "")
        if base_url and api_key:
            return Found(base_url=base_url, api_key=api_key, source=path)
    return None
