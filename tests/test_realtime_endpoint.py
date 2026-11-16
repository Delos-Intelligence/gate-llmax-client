"""``LLMClient.realtime_endpoint``: the base URL and billing headers a vendor's own realtime client sends to Gate."""

from __future__ import annotations

from gate_llmax import LLMClient
from gate_llmax.models.byok import ByokCredential, ByokPlan


def test_endpoint_carries_key_operation_and_session() -> None:
    client = LLMClient(api_key="k", base_url="https://gate.example/")

    endpoint = client.realtime_endpoint(operation="call", session_id="s1")

    assert endpoint.base_url == "https://gate.example/v1/realtime"
    assert endpoint.headers["X-Gate-Key"] == "k"
    assert endpoint.headers["X-Gate-Operation"] == "call"
    assert endpoint.headers["X-Gate-Session-Id"] == "s1"
    assert "X-Gate-Plan" not in endpoint.headers
    assert "X-Gate-Byok-Plan" not in endpoint.headers


def test_operation_takes_the_client_prefix() -> None:
    client = LLMClient(api_key="k", base_url="http://x").prefix_operation("voice")

    assert client.realtime_endpoint(operation="call").headers["X-Gate-Operation"] == "voice/call"


def test_plan_and_byok_plan_ride_as_headers() -> None:
    byok = ByokPlan(credential=ByokCredential(api_key="upstream"), models=["gpt-live-1"])
    client = LLMClient(api_key="k", base_url="http://x", default_plan="omicron", byok_plan=byok)

    headers = client.realtime_endpoint(operation="call").headers

    assert headers["X-Gate-Plan"] == "omicron"
    assert headers["X-Gate-Byok-Plan"] == byok.to_header()
    assert "X-Gate-Session-Id" not in headers


def test_each_endpoint_gets_its_own_call_id() -> None:
    client = LLMClient(api_key="k", base_url="http://x")

    first = client.realtime_endpoint(operation="call").headers["X-Gate-Call-Id"]
    second = client.realtime_endpoint(operation="call").headers["X-Gate-Call-Id"]

    assert first != second
