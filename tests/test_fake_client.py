"""``FakeClient`` returns canned replies through the full builder API with no network."""

from __future__ import annotations

import asyncio

from gate_llmax import FakeClient


def test_fake_call_returns_canned_reply() -> None:
    client = FakeClient("hello world")
    resp = asyncio.run(client.request(prompt="hi", operation="test_fake_client").call("m"))
    assert resp.raw_text == "hello world"
    assert resp.model == "m"


def test_fake_callable_reply_sees_request() -> None:
    client = FakeClient(lambda req: f"echo:{req.operation}")
    resp = asyncio.run(client.request(prompt="hi", operation="test_fake_client").call("m"))
    assert resp.raw_text == "echo:test_fake_client"


def test_fake_cast_json_parses_canned_json() -> None:
    client = FakeClient('{"a": 1}')
    resp = asyncio.run(client.request(prompt="hi", operation="test_fake_client").cast_json().call("m"))
    assert resp.json_response == {"a": 1}


def test_fake_call_stream_emits_words() -> None:
    client = FakeClient("one two three")

    async def drain() -> str:
        return "".join([c.text async for c in client.request(prompt="hi", operation="test_fake_client").call_stream("m")])

    assert asyncio.run(drain()) == "one two three"
