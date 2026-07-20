"""Gate agent tooling: the ``gate-llmax`` CLI, the Gate MCP server, and the plan-fallback helper.

Install into a project with ``gate-llmax agent install``; the MCP server is ``gate-llmax agent mcp``.
"""

from gate_llmax.agent.prefer import PreferResult, PreferStep, build_prefer_list

__all__ = ["PreferResult", "PreferStep", "build_prefer_list"]
