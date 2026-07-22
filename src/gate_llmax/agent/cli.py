"""``gate-llmax`` CLI — install the Gate agent skill + MCP into a project, or run the MCP server.

    gate-llmax agent install [--project DIR] [--force]   copy the skill + register the MCP
    gate-llmax agent mcp                                  run the MCP server over stdio
    gate-llmax agent uninstall [--project DIR]            remove the skill + MCP entry
    gate-llmax heavy-test MODEL [-n N] [--rate R]         hammer a chat model, print the report

``install`` writes ``<project>/.claude/skills/gate-llmax/SKILL.md``, adds a ``gate-llmax`` server
to ``<project>/.mcp.json`` (preserving any other servers), then settles the gateway credentials:
it reads them from the project's own env files when they are there (any prefix, e.g.
``COSMOS_GATE_API_KEY``) and otherwise prompts. Either way it verifies them against the gateway
and stores them outside the project, so no secret lands in ``.mcp.json`` and nothing has to be
exported by hand. Run it from the consumer project, e.g. ``uv run gate-llmax agent install``.

``heavy-test`` is the MCP ``heavy_test`` tool on the command line — same suite, same report, as
JSON on stdout. It uses the same credentials and spends real quota.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.resources
import json
import sys
from pathlib import Path

from gate_llmax.agent import credentials, env_discovery

_MCP_SERVER_KEY = "gate-llmax"


def _skill_text() -> str:
    """The packaged SKILL.md content."""
    return (importlib.resources.files("gate_llmax.agent") / "skill" / "SKILL.md").read_text(encoding="utf-8")


def _mcp_server_entry() -> dict:
    """The ``.mcp.json`` entry that launches this MCP server.

    No secrets here: the server reads GATE_BASE_URL / GATE_API_KEY from its own environment, and
    falls back to the credentials file this command prompts for.
    """
    return {
        "command": "uv",
        "args": ["run", "gate-llmax", "agent", "mcp"],
    }


def _probe(base_url: str, api_key: str) -> str | None:
    """Call the gateway once; return an error message, or None when the credentials work."""
    import asyncio

    from gate_llmax.client import LLMClient

    async def run() -> None:
        async with LLMClient(api_key=api_key, base_url=base_url, timeout=30) as client:
            await client.list_models()

    try:
        asyncio.run(run())
    except Exception as exc:  # any failure is reported verbatim
        return str(exc)
    return None


def _verify_and_save(base_url: str, api_key: str, *, interactive: bool) -> None:
    """Probe the gateway with these credentials, then store them for the MCP server to pick up."""
    error = _probe(base_url, api_key)
    if error:
        print(f"⚠ the gateway rejected these credentials: {error}")
        if interactive and input("  save them anyway? [y/N]: ").strip().lower() not in {"y", "yes"}:
            print("• not saved.")
            return
    else:
        print("✓ credentials verified against the gateway")

    path = credentials.save(base_url, api_key)
    print(f"✓ saved   {path}")


def _configure_credentials(project: Path, *, prompt: bool) -> None:
    """Store the gateway credentials, reusing the project's own env files when they carry them."""
    interactive = prompt and sys.stdin.isatty()

    found = env_discovery.discover(project)
    if found:
        print(f"\n✓ found  {found.base_url} + an API key in {found.source.relative_to(project)}")
        if not interactive or input("  use them? [Y/n]: ").strip().lower() not in {"n", "no"}:
            _verify_and_save(found.base_url, found.api_key, interactive=interactive)
            return

    if not interactive:
        print("\n• no gateway credentials in the project's env files; export GATE_BASE_URL and GATE_API_KEY instead.")
        return

    print()
    _prompt_credentials()


def _prompt_credentials() -> None:
    """Ask for the gateway URL and key, then store them for the MCP server to pick up."""
    import getpass

    current_url, current_key = credentials.resolve()

    suffix = f" [{current_url}]" if current_url else ""
    base_url = input(f"Gate base URL{suffix}: ").strip() or current_url
    if not base_url:
        print("• skipped: no base URL given; the MCP server will have no gateway to talk to.")
        return

    suffix = " [keep existing]" if current_key else ""
    api_key = getpass.getpass(f"Gate API key (dev key unlocks the plan tools){suffix}: ").strip() or current_key
    if not api_key:
        print("• skipped: no API key given; the MCP server will have no credentials.")
        return

    _verify_and_save(base_url, api_key, interactive=True)


def _install(project: Path, *, force: bool, prompt: bool) -> int:
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

    _configure_credentials(project, prompt=prompt)

    print("\nNext steps:")
    if not have_mcp:
        print('  1. Add the MCP runtime dependency:  uv add "gate-llmax[agent]"')
    print(f"  {'2' if not have_mcp else '1'}. Restart Claude Code (or run /mcp) so it picks up the 'gate-llmax' server.")
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


def _heavy_test(args: argparse.Namespace) -> int:
    """Run the heavy-test suite against one model and print the JSON report."""
    import asyncio

    from gate_llmax.agent.heavy_test import run_heavy_test
    from gate_llmax.client import LLMClient

    base_url, api_key = credentials.resolve()
    if not base_url or not api_key:
        print("error: no gateway config. Run `gate-llmax agent install`, or export GATE_BASE_URL / GATE_API_KEY.", file=sys.stderr)
        return 1

    async def run() -> dict:
        async with LLMClient(api_key=api_key, base_url=base_url, timeout=args.timeout) as client:
            return await run_heavy_test(
                client,
                args.model,
                n=args.n,
                rate=args.rate,
                plan=args.plan,
                only=args.only,
                include_runs=args.include_runs,
            )

    report = asyncio.run(run())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``gate-llmax`` console script."""
    parser = argparse.ArgumentParser(prog="gate-llmax", description="Gate LLM gateway client tooling.")
    sub = parser.add_subparsers(dest="group", required=True)

    agent = sub.add_parser("agent", help="Agent tooling: install the skill + MCP, or run the MCP server.")
    actions = agent.add_subparsers(dest="action", required=True)

    p_install = actions.add_parser("install", help="Install the Gate skill + MCP into a project.")
    p_install.add_argument("--project", default=".", help="Project directory to install into (default: cwd).")
    p_install.add_argument("--force", action="store_true", help="Overwrite an existing skill without the notice.")
    p_install.add_argument(
        "--no-prompt",
        action="store_true",
        help="Never ask; take credentials from the project env files or leave them unset.",
    )

    actions.add_parser("mcp", help="Run the Gate MCP server over stdio.")

    p_uninstall = actions.add_parser("uninstall", help="Remove the Gate skill + MCP from a project.")
    p_uninstall.add_argument("--project", default=".", help="Project directory to clean (default: cwd).")

    heavy = sub.add_parser("heavy-test", help="Hammer a chat model with every request shape it can serve.")
    heavy.add_argument("model", help="Chat model name as registered on the gateway.")
    heavy.add_argument("-n", type=int, default=5, help="How many times to replay the suite (default: 5).")
    heavy.add_argument("--rate", type=float, default=6.0, help="Launch rate in requests per minute (default: 6).")
    heavy.add_argument("--plan", default=None, help="Hosting plan to route under.")
    heavy.add_argument("--only", nargs="*", default=None, help="Restrict to these case ids or tags.")
    heavy.add_argument("--timeout", type=int, default=240, help="Per-request timeout in seconds (default: 240).")
    heavy.add_argument("--include-runs", action="store_true", help="Include every individual run in the report.")

    args = parser.parse_args(argv)

    if args.group == "heavy-test":
        return _heavy_test(args)

    if args.group == "agent":
        if args.action == "install":
            return _install(Path(args.project), force=args.force, prompt=not args.no_prompt)
        if args.action == "mcp":
            return _run_mcp()
        if args.action == "uninstall":
            return _uninstall(Path(args.project))
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
