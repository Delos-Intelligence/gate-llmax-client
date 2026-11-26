"""The MCP catalogue tools read and edit connector rows within the guard rails the gate admin keeps."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gate_llmax.agent import mcp_server
from gate_llmax.client import LLMClient

BASE_URL = "https://gate.test"
_ROW = {
    "id": "row-1",
    "slug": "pipecorn",
    "name": "Pipecorn",
    "type": "builtin",
    "enabled": True,
    "icon": "data:image/png;base64,AAAA",
    "category_id": "cat-1",
    "description": "Find a B2B contact's phone.",
    "instructions": "Call single_enrich_contact with the LinkedIn URL.",
    "config": {
        "url": "https://mcp.pipecorn.com/mcp",
        "api_key_header": "X-API-Key",
        "oauth": {"client_secret": "s3cr3t", "token_url": "https://t"},
    },
}
_CATEGORY = {"id": "cat-1", "slug": "sales", "name": "Sales", "sort_order": 1}


def gateway(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]] | None = None,
    *,
    icon_status: int | None = 200,
    spec_builds: bool = False,
) -> list[tuple[str, str, Any]]:
    """Point the tools at a fake gate holding ``rows``; returns every (method, path, body) it was sent."""
    sent: list[tuple[str, str, Any]] = []
    stored = rows if rows is not None else [dict(_ROW)]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        sent.append((request.method, request.url.path, body))
        path = request.url.path
        if path.endswith("/fetch-icon"):
            if icon_status is None:
                raise httpx.ReadTimeout("the vendor's CDN did not answer")
            return httpx.Response(icon_status, json={"detail": "not a PNG or SVG"})
        if path == "/admin/mcp/servers/validate":
            return httpx.Response(200, json={"ok": spec_builds, "error": None if spec_builds else "no paths"})
        if path in {"/admin/mcp/servers", "/v1/mcp/servers"} and request.method == "GET":
            return httpx.Response(200, json=stored)
        if path == "/admin/mcp/categories":
            return httpx.Response(200, json=[_CATEGORY])
        if request.method == "POST" and path == "/admin/mcp/servers":
            return httpx.Response(200, json={"id": "row-2", **body})
        return httpx.Response(200, json=body or {})

    def client() -> LLMClient:
        http = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
        return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=http)

    monkeypatch.setattr(mcp_server, "_client", client)
    return sent


def _writes(sent: list[tuple[str, str, Any]]) -> list[tuple[str, str, Any]]:
    return [call for call in sent if call[0] in {"POST", "PATCH"}]


async def test_the_listing_keeps_its_documented_fields_without_icons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base64 icons made the whole catalogue 3.4 MB, too big for an agent to read."""
    gateway(monkeypatch)
    [row] = await mcp_server.list_mcp_servers()
    assert set(row) == {"slug", "name", "type", "enabled", "auth_type", "url", "description", "has_delos_key", "delos_key_for"}


async def test_a_json_row_shows_its_spec_summed_up_and_a_listed_secret_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full OpenAPI spec would cost the agent tens of thousands of tokens for one read."""
    spec = {"info": {"title": "Acme API"}, "paths": {"/a": {}, "/b": {}}}
    config = {"openapi": spec, "private_keys": ["k1", "k2"], "is_members_own_key": True}
    gateway(monkeypatch, [{**_ROW, "type": "json", "config": config}])
    shown = (await mcp_server.get_mcp_server("pipecorn"))["config"]
    assert shown == {"openapi": {"title": "Acme API", "paths": 2}, "private_keys": "***", "is_members_own_key": True}


async def test_a_row_is_read_with_its_secrets_masked_and_its_icon_as_a_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway(monkeypatch)
    row = await mcp_server.get_mcp_server("pipecorn")
    assert row["config"]["oauth"] == {"client_secret": "***", "token_url": "https://t"}
    assert row["config"]["api_key_header"] == "X-API-Key"
    assert (row["has_icon"], "icon" in row) == (True, False)


async def test_a_new_row_is_created_disabled_on_its_category(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = gateway(monkeypatch)
    out = await mcp_server.create_mcp_server(
        "acme", "Acme", "builtin", "Acme CRM.", "Use search first.", "sales", url="https://mcp.acme.com/mcp", auth_type="oauth"
    )
    [(_, path, body)] = _writes(sent)
    assert (path, body["enabled"], body["category_id"]) == ("/admin/mcp/servers", False, "cat-1")
    assert body["config"] == {"url": "https://mcp.acme.com/mcp", "auth_type": "oauth"}
    assert out["enabled"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        {"url": "https://mcp.acme.com/mcp", "auth_type": "api_key", "config": {"delos_key_credential_id": "cred"}},
        {"url": "https://mcp.acme.com/mcp", "auth_type": "api_key", "config": {"delos_key_for": ["sophie"]}},
        {"url": "https://mcp.acme.com/mcp", "auth_type": "oauth", "oauth": {"client_secret": "s"}},
        {"url": "https://mcp.acme.com/mcp", "auth_type": "oauth", "config": {"oauth": {"client_secret": "s"}}},
        {"auth_type": "oauth"},
    ],
)
async def test_a_creation_the_admin_keeps_or_that_lacks_its_endpoint_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    sent = gateway(monkeypatch)
    out = await mcp_server.create_mcp_server("acme", "Acme", "builtin", "Acme CRM.", "Use search first.", "sales", **arguments)
    assert "error" in out
    assert _writes(sent) == []


async def test_a_json_row_whose_spec_does_not_build_is_not_created(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vendor's `/oauth/token` route or `password` field is its spec, not one of our secrets: the gate judges the spec."""
    sent = gateway(monkeypatch)
    spec = {"paths": {"/oauth/token": {}}, "components": {"schemas": {"Login": {"properties": {"password": {}}}}}}
    out = await mcp_server.create_mcp_server(
        "acme",
        "Acme",
        "json",
        "Acme CRM.",
        "Use search first.",
        "sales",
        config={"base_url": "https://api.acme.com", "openapi": spec},
    )
    assert "error" in out
    assert [path for _, path, _ in _writes(sent)] == ["/admin/mcp/servers/validate"]


async def test_a_key_without_mcp_edit_is_told_which_flag_it_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate answers 403 to a dev key on /admin/mcp: the agent must learn it needs the mcp_edit flag."""

    def client() -> LLMClient:
        refuse = httpx.MockTransport(lambda _: httpx.Response(403, json={"detail": "Forbidden"}))
        return LLMClient(api_key="k", base_url=BASE_URL, httpx_aclient=httpx.AsyncClient(base_url=BASE_URL, transport=refuse))

    monkeypatch.setattr(mcp_server, "_client", client)
    out = await mcp_server.get_mcp_server("pipecorn")
    assert "mcp_edit" in out["error"]


async def test_appended_instructions_keep_the_current_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = gateway(monkeypatch)
    await mcp_server.update_mcp_server("pipecorn", append_instructions="A 422 live-mode refusal needs live: true.")
    [(method, path, body)] = _writes(sent)
    assert (method, path) == ("PATCH", "/admin/mcp/servers/row-1")
    assert body["instructions"] == "Call single_enrich_contact with the LinkedIn URL.\n\nA 422 live-mode refusal needs live: true."


@pytest.mark.parametrize(
    "arguments",
    [
        {"config": {"delos_key_for": ["sophie"]}},
        {"config": {"auto_connect_all": True}},
        {"config": {"suggested_for": ["sophie"]}},
        {"config": {"oauth": {"client_secret": "x"}}},
        {"config": {"headers": {"Api-Version": "1"}}},
        {"config": {"oauth": {"delos_auto_connect": "https://issuer"}}},
        {"config": {"oauth": {"sources": [{"client_secret": "x"}]}}},
        {"instructions": "new", "append_instructions": "more"},
        {"category": "unknown"},
        {"description": "  "},
    ],
)
async def test_an_update_the_admin_keeps_or_that_cannot_apply_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    """Checking a row for a template reaches every worker already hired: that stays the admin's call."""
    sent = gateway(monkeypatch)
    out = await mcp_server.update_mcp_server("pipecorn", **arguments)
    assert "error" in out
    assert _writes(sent) == []


async def test_a_partial_oauth_change_keeps_the_other_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate merges config one level deep: sent alone, a scope would wipe the authorize and token URLs."""
    sent = gateway(monkeypatch)
    await mcp_server.update_mcp_server("pipecorn", config={"oauth": {"scope": "read"}})
    [(_, _, body)] = _writes(sent)
    assert body["config"]["oauth"] == {"client_secret": "s3cr3t", "token_url": "https://t", "scope": "read"}


async def test_a_new_spec_replaces_the_stored_one_once_it_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Merged, a route the new spec dropped would stay a live tool for every org."""
    row = {**_ROW, "type": "json", "config": {"base_url": "https://api.acme.com", "openapi": {"paths": {"/a": {}, "/b": {}}}}}
    sent = gateway(monkeypatch, [row], spec_builds=True)
    await mcp_server.update_mcp_server("pipecorn", config={"openapi": {"paths": {"/a": {}}}})
    assert [path for _, path, _ in _writes(sent)] == ["/admin/mcp/servers/validate", "/admin/mcp/servers/row-1"]
    assert _writes(sent)[-1][2]["config"]["openapi"] == {"paths": {"/a": {}}}


@pytest.mark.parametrize("icon_status", [415, None])
async def test_a_logo_the_gate_refuses_still_reports_the_created_row(monkeypatch: pytest.MonkeyPatch, icon_status: int | None) -> None:
    """Reported as a failure, the creation was retried and refused on its own slug."""
    sent = gateway(monkeypatch, icon_status=icon_status)
    out = await mcp_server.create_mcp_server(
        "acme",
        "Acme",
        "builtin",
        "Acme CRM.",
        "Use search first.",
        "sales",
        url="https://mcp.acme.com/mcp",
        auth_type="oauth",
        icon_url="https://acme.com/logo.jpg",
    )
    assert (out["created"], "icon_error" in out) == ("acme", True)
    assert _writes(sent)[-1] == ("PATCH", "/admin/mcp/servers/row-2", {"icon": None})


async def test_a_row_its_client_serves_is_enabled_without_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's card rule exempts it: its client tells its agent in its own words."""
    sent = gateway(monkeypatch, [{**_ROW, "enabled": False, "instructions": None, "config": {"served_by": "client"}}])
    await mcp_server.enable_mcp_server("pipecorn")
    assert _writes(sent) == [("PATCH", "/admin/mcp/servers/row-1", {"enabled": True})]


async def test_a_row_without_an_icon_is_not_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's CI refuses an enabled row without icon, shelf, description or instructions."""
    sent = gateway(monkeypatch, [{**_ROW, "enabled": False, "icon": None}])
    out = await mcp_server.enable_mcp_server("pipecorn")
    assert "icon" in out["error"]
    assert _writes(sent) == []


async def test_disabling_switches_the_row_off(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = gateway(monkeypatch)
    out = await mcp_server.disable_mcp_server("pipecorn")
    assert _writes(sent) == [("PATCH", "/admin/mcp/servers/row-1", {"enabled": False})]
    assert out["enabled"] is False
