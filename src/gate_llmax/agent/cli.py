"""``gate-llmax`` CLI — install the Gate agent skill + MCP into a project, or run the MCP server.

    gate-llmax agent install [--project DIR] [--force]   copy the skill + register the MCP
    gate-llmax agent mcp                                  run the MCP server over stdio
    gate-llmax agent uninstall [--project DIR]            remove the skill + MCP entry

``install`` writes ``<project>/.claude/skills/gate-llmax/SKILL.md`` and adds a ``gate-llmax``
server to ``<project>/.mcp.json`` (preserving any other servers). Run it from the consumer
project, e.g. ``uv run gate-llmax agent install``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.resources
import json
import sys
from pathlib import Path

_MCP_SERVER_KEY = "gate-llmax"


def _skill_text() -> str:
    """The packaged SKILL.md content."""
    return (importlib.resources.files("gate_llmax.agent") / "skill" / "SKILL.md").read_text(encoding="utf-8")


def _mcp_server_entry() -> dict:
    """The ``.mcp.json`` entry that launches this MCP server.

    ``${VAR}`` values are expanded by Claude Code from the environment, so no secrets are written
    into the file — set GATE_BASE_URL and GATE_API_KEY (a dev key) in your shell / .env.
    """
    return {
        "command": "uv",
        "args": ["run", "gate-llmax", "agent", "mcp"],
        "env": {
            "GATE_BASE_URL": "${GATE_BASE_URL}",
            "GATE_API_KEY": "${GATE_API_KEY}",
        },
    }


def _install(project: Path, *, force: bool) -> int:
    project = project.resolve()
    if not project.is_dir():
        print(f"error: project directory does not exist: {project}", file=sys.stderr)
        return 1

    skill_dir = project / ".claude" / "skills" / _MCP_SERVER_KEY
    skill_file = skill_dir / "SKILL.md"
    skill_dir.mkdir(parents=True, exist_ok=True)
    existed = skill_file.exists()
    if existed and not force:
        print(f"• skill already present, overwriting: {skill_file}")
    skill_file.write_text(_skill_text(), encoding="utf-8")
    print(f"{'↻ updated' if existed else '✓ wrote'} skill  {skill_file.relative_to(project)}")

    mcp_path = project / ".mcp.json"
    if mcp_path.exists():
        try:
            config = json.loads(mcp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"error: {mcp_path} is not valid JSON ({exc}); fix or move it and retry.", file=sys.stderr)
            return 1
        if not isinstance(config, dict):
            print(f"error: {mcp_path} must contain a JSON object.", file=sys.stderr)
            return 1
    else:
        config = {}

    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        print(f"error: {mcp_path} has a non-object 'mcpServers'.", file=sys.stderr)
        return 1
    updated = _MCP_SERVER_KEY in servers
    servers[_MCP_SERVER_KEY] = _mcp_server_entry()
    mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"{'↻ updated' if updated else '✓ added'} MCP    {mcp_path.relative_to(project)}  (server '{_MCP_SERVER_KEY}')")

    try:
        import mcp  # noqa: F401  (probe only)

        have_mcp = True
    except ModuleNotFoundError:
        have_mcp = False

    print("\nNext steps:")
    if not have_mcp:
        print('  1. Add the MCP runtime dependency:  uv add "gate-llmax[agent]"')
    print(f"  {'2' if not have_mcp else '1'}. Export your gateway config (a dev key unlocks the plan tools):")
    print("       export GATE_BASE_URL=https://your-gate-instance")
    print("       export GATE_API_KEY=your-dev-key")
    print(f"  {'3' if not have_mcp else '2'}. Restart Claude Code (or run /mcp) so it picks up the 'gate-llmax' server.")
    return 0


def _uninstall(project: Path) -> int:
    project = project.resolve()
    skill_file = project / ".claude" / "skills" / _MCP_SERVER_KEY / "SKILL.md"
    if skill_file.exists():
        skill_file.unlink()
        with contextlib.suppress(OSError):
            skill_file.parent.rmdir()
        print(f"✓ removed skill  {skill_file.relative_to(project)}")
    else:
        print("• no skill to remove")

    mcp_path = project / ".mcp.json"
    if mcp_path.exists():
        try:
            config = json.loads(mcp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"• left {mcp_path.name} untouched (not valid JSON)")
            return 0
        servers = config.get("mcpServers") if isinstance(config, dict) else None
        if isinstance(servers, dict) and servers.pop(_MCP_SERVER_KEY, None) is not None:
            mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"✓ removed MCP server '{_MCP_SERVER_KEY}' from {mcp_path.name}")
        else:
            print(f"• no '{_MCP_SERVER_KEY}' server in {mcp_path.name}")
    return 0


def _run_mcp() -> int:
    from gate_llmax.agent.mcp_server import main as run_server

    run_server()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``gate-llmax`` console script."""
    parser = argparse.ArgumentParser(prog="gate-llmax", description="Gate LLM gateway client tooling.")
    sub = parser.add_subparsers(dest="group", required=True)

    agent = sub.add_parser("agent", help="Agent tooling: install the skill + MCP, or run the MCP server.")
    actions = agent.add_subparsers(dest="action", required=True)

    p_install = actions.add_parser("install", help="Install the Gate skill + MCP into a project.")
    p_install.add_argument("--project", default=".", help="Project directory to install into (default: cwd).")
    p_install.add_argument("--force", action="store_true", help="Overwrite an existing skill without the notice.")

    actions.add_parser("mcp", help="Run the Gate MCP server over stdio.")

    p_uninstall = actions.add_parser("uninstall", help="Remove the Gate skill + MCP from a project.")
    p_uninstall.add_argument("--project", default=".", help="Project directory to clean (default: cwd).")

    args = parser.parse_args(argv)

    if args.group == "agent":
        if args.action == "install":
            return _install(Path(args.project), force=args.force)
        if args.action == "mcp":
            return _run_mcp()
        if args.action == "uninstall":
            return _uninstall(Path(args.project))
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
