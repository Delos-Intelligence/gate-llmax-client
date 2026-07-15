"""Configuration/admin models shared between backend and client."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DeploymentStatus(StrEnum):
    """Lifecycle / health status of a provider deployment."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ERRORING = "ERRORING"
    RATE_LIMITED = "RATE_LIMITED"


class ModelPurpose(StrEnum):
    """What kind of operation a model is designed for."""

    CHAT = "chat"
    EMBED = "embed"
    AUDIO = "audio"
    VISION = "vision"
    IMAGES = "images"
    TTS = "tts"
    AUDIO_ISOLATION = "audio_isolation"
    DUBBING = "dubbing"
    VIDEO = "video"


class ModelCapabilities(BaseModel):
    """Capability flags for a registered LLM model."""

    supports_tools: bool = False
    supports_images: bool = False
    supports_reasoning: bool = False


class ModelInfo(BaseModel):
    """Public metadata for a registered model."""

    id: str
    name: str
    purpose: ModelPurpose = ModelPurpose.CHAT
    capabilities: ModelCapabilities
    input_token_price: float = Field(description="USD per 1M input tokens")
    output_token_price: float = Field(description="USD per 1M output tokens")
    input_cache_price: float = Field(description="USD per 1M cached input tokens", default=0.0)
    max_output_tokens: int | None = None
    max_tries: int = 2
    timeout: int = 120
    extra_attributes: dict[str, Any] = Field(default_factory=dict, description="Arbitrary per-model attributes (e.g. selection weights).")
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ExtraAttributeName(BaseModel):
    """A registered extra-attribute name (controlled vocabulary for model `extra_attributes`)."""

    id: str
    name: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DeploymentInfo(BaseModel):
    """Public metadata for a model deployment."""

    id: str
    model_id: str
    name: str
    api_provider: str
    hosting_provider: str = ""
    priority: int
    region: str | None = None
    country: str | None = None
    status: DeploymentStatus = DeploymentStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ResolvedDeployment(BaseModel):
    """One candidate deployment a model would route to (as seen by ``/v1/resolve``)."""

    id: str
    name: str
    api_provider: str
    hosting_provider: str = ""
    region: str | None = None
    country: str | None = None
    priority: int
    selected: bool = Field(default=False, description="True if the call would hit this deployment (only set when a pin key is given).")


class ResolveResponse(BaseModel):
    """What a chat call to a given model would resolve to, without making the call."""

    requested_model: str
    resolved_model: str = Field(description="The concrete model after redirect/alias resolution.")
    redirect_from: str | None = Field(default=None, description="Alias name if a model_redirect was hit; None if a real model matched.")
    purpose: ModelPurpose = ModelPurpose.CHAT
    deployment_count: int
    selection_strategy: Literal["pinned", "round_robin"] = Field(
        description="'pinned' when a seed_routing/session_id deterministically selects one deployment; 'round_robin' otherwise.",
    )
    candidates: list[ResolvedDeployment] = Field(description="All active deployments after zone filtering, ordered by priority.")
    selected: ResolvedDeployment | None = Field(
        default=None,
        description="The deployment the call would hit, when deterministic (pin key given). None under round-robin (picked per-call).",
    )
    extra_attributes: dict[str, Any] = Field(default_factory=dict, description="The resolved model's arbitrary attributes.")
