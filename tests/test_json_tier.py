"""The JSON request/response tier: ``.cast_json()`` and its typed extension ``.cast(T)``.

Pure-logic tests over the builder ``_finalize`` hooks — no live gateway.
"""

from __future__ import annotations

from pydantic import BaseModel

from gate_llmax import JsonLLMResponse, JsonRequestBuilder, LLMResponse, TypedLLMResponse
from gate_llmax.request import CastedRequestBuilder
from gate_llmax.types import OutputStatus


class Weather(BaseModel):
    """Sample target model."""

    city: str
    temp: int


def _ok(raw: str) -> LLMResponse:
    return LLMResponse(raw_text=raw, status=OutputStatus.SUCCESS)


def test_cast_json_parses_into_json_response() -> None:
    """`.cast_json()` finalizes into a JsonLLMResponse with `json_response` set (fences stripped)."""
    out = JsonRequestBuilder.model_construct(client=None)._finalize(_ok('```json\n{"city":"Paris","temp":20}\n```'))
    assert isinstance(out, JsonLLMResponse)
    assert out.json_response == {"city": "Paris", "temp": 20}


def test_cast_extends_json_with_typed_value() -> None:
    """`.cast(T)` is a json request too: json_response AND the validated value, and IS-A JsonLLMResponse."""
    out = CastedRequestBuilder.model_construct(client=None, model_type=Weather)._finalize(_ok('{"city":"Lyon","temp":18}'))
    assert isinstance(out, TypedLLMResponse)
    assert isinstance(out, JsonLLMResponse)
    assert out.json_response == {"city": "Lyon", "temp": 18}
    assert out.value == Weather(city="Lyon", temp=18)


def test_unparseable_text_yields_none() -> None:
    """Non-JSON text leaves json_response/value as None without raising."""
    j = JsonRequestBuilder.model_construct(client=None)._finalize(_ok("just prose, no json"))
    assert j.json_response is None
    c = CastedRequestBuilder.model_construct(client=None, model_type=Weather)._finalize(_ok("nope"))
    assert c.json_response is None
    assert c.value is None


def test_server_json_object_is_preferred() -> None:
    """A server-set json_object (e.g. vision OCR) flows into json_response without re-parsing raw_text."""
    resp = LLMResponse(raw_text="", status=OutputStatus.SUCCESS, json_object={"server": True})
    out = JsonRequestBuilder.model_construct(client=None)._finalize(resp)
    assert out.json_response == {"server": True}


def test_cast_json_and_cast_force_response_format() -> None:
    """Both `.cast_json()` and `.cast(T)` set response_format so providers force valid JSON."""
    from gate_llmax.request import RequestBuilder

    base = RequestBuilder.model_construct(client=None, response_format=None)
    assert base.cast_json().response_format == {"type": "json_object"}
    assert base.cast(Weather).response_format == {"type": "json_object"}
