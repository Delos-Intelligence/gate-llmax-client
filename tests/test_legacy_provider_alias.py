"""Legacy ``provider`` wire alias: pre-v0.3 clients keep parsing v0.3 gateway responses."""

from __future__ import annotations

from pydantic import BaseModel

from gate_llmax.models.config import DeploymentInfo, ResolvedDeployment
from gate_llmax.models.response import RawUsage, StreamChunk


class OldResolvedDeployment(BaseModel):
    """The pre-v0.3 client shape: ``provider`` is a required field."""

    id: str
    name: str
    provider: str
    priority: int


def test_provider_alias_serialized_everywhere() -> None:
    assert RawUsage(api_provider="openai").model_dump()["provider"] == "openai"
    assert StreamChunk(api_provider="openai").model_dump()["provider"] == "openai"
    dep = ResolvedDeployment(id="d", name="n", api_provider="openai", hosting_provider="azure", priority=1)
    assert dep.model_dump()["provider"] == "openai"
    info = DeploymentInfo(id="d", model_id="m", name="n", api_provider="openai", priority=1)
    assert info.model_dump()["provider"] == "openai"


def test_pre_v03_client_parses_v03_resolve_payload() -> None:
    payload = ResolvedDeployment(id="d", name="n", api_provider="openai", hosting_provider="azure", priority=1).model_dump()
    old = OldResolvedDeployment.model_validate(payload)
    assert old.provider == "openai"


def test_provider_attribute_reads_keep_working() -> None:
    """cosmos-style ``.provider`` reads resolve to ``api_provider``."""
    assert RawUsage(api_provider="openai").provider == "openai"
    assert StreamChunk().provider is None
