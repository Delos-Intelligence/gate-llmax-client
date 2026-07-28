"""Configuration/admin models shared between backend and client."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


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
    developer_id: str | None = None
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


class PlanInfo(BaseModel):
    """A hosting plan: a named preset of hosting providers (a cost/infra tier, e.g. ``omicron``)."""

    id: str
    name: str
    description: str | None = None
    sort_order: int = 0


class ModelPlanRow(BaseModel):
    """One model and the plans it is reachable on (has a deployment on an admitted host).

    ``available_plan_ids`` lists the plans a chat/… call to ``model_name`` can succeed under
    (``plan=`` filters routing to that plan's hosting providers). Empty ⇒ reachable on no plan.
    """

    model_id: str
    model_name: str
    purpose: ModelPurpose = ModelPurpose.CHAT
    developer_id: str | None = None
    available_plan_ids: list[str] = Field(default_factory=list)


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

    @computed_field
    @property
    def provider(self) -> str:
        """Deprecated alias of ``api_provider`` — kept on the wire for pre-v0.3 clients."""
        return self.api_provider


class ResolvedDeployment(BaseModel):
    """One candidate deployment a model would route to (as seen by ``/v1/resolve``)."""

    id: str
    name: str
    api_provider: str
    hosting_provider: str = ""
    region: str | None = None
    country: str | None = None
    priority: int
    status: str = "ACTIVE"
    last_error: str | None = Field(default=None, description="Most recent failing-ping error text; None when healthy.")
    selected: bool = Field(default=False, description="True if the call would hit this deployment (only set when a pin key is given).")

    @computed_field
    @property
    def provider(self) -> str:
        """Deprecated alias of ``api_provider`` — kept on the wire for pre-v0.3 clients."""
        return self.api_provider


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
    all_deployments: list[ResolvedDeployment] = Field(
        default_factory=list,
        description="Every deployment regardless of status (ERROR/INACTIVE/etc) — shows why a model has no routable candidates.",
    )
    selected: ResolvedDeployment | None = Field(
        default=None,
        description="The deployment the call would hit, when deterministic (pin key given). None under round-robin (picked per-call).",
    )
    extra_attributes: dict[str, Any] = Field(default_factory=dict, description="The resolved model's arbitrary attributes.")
