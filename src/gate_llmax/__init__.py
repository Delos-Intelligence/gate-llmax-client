"""Gate Client SDK ergonomic Python interface to the Gate LLM Gateway."""

from importlib.metadata import PackageNotFoundError, version

from gate_llmax.models.audio import AudioRequest, AudioResponse, TranscriptionSegment
from gate_llmax.models.audio_gen import AudioGenMode, AudioGenRequest, AudioGenResponse, AudioMode, DialogueTurn
from gate_llmax.models.audio_isolation import AudioIsolationRequest, AudioIsolationResponse
from gate_llmax.models.config import (
    DeploymentInfo,
    DeploymentStatus,
    ExtraAttributeName,
    FallbackRung,
    ModelCapabilities,
    ModelInfo,
    ModelPlanRow,
    ModelPurpose,
    PlanInfo,
    ResolvedDeployment,
    ResolveResponse,
)
from gate_llmax.models.dubbing import DubbingRequest, DubbingResponse
from gate_llmax.models.embed import EmbedObject, EmbedRequest, EmbedResponse
from gate_llmax.models.images import ImageData, ImageRequest, ImageResponse
from gate_llmax.models.messages import ImageMessage, Message, MessageRole, TextMessage
from gate_llmax.models.request import BestTarget, CallControl, RequestSpecifics, ResolveRequest, ZoneSelection
from gate_llmax.models.response import (
    BaseAudioResponse,
    LLMCallRecord,
    LLMResponse,
    MulticallStreamFrame,
    RawUsage,
    StreamChunk,
    ToolCall,
    ToolFunction,
    VisionLLMResponse,
)
from gate_llmax.models.responses import ResponsesRequest, ResponsesResponse
from gate_llmax.models.tts import TTSRequest, TTSResponse
from gate_llmax.models.video import VideoRequest, VideoResponse
from gate_llmax.models.vision import VisionLine, VisionOCR, VisionOCRRequest, VisionPoint, VisionWord
from gate_llmax.types import JsonDict, JsonValue, OutputStatus, ReasoningEffort

from .client import LLMClient
from .exceptions import (
    LLMAuthError,
    LLMBudgetError,
    LLMCapabilityError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMContextOverflowError,
    LLMError,
    LLMEscapeHatchWarning,
    LLMModelNotFoundError,
    LLMServerError,
    LLMTimeoutError,
)
from .fake import FakeClient
from .ratelimit import RateLimit
from .request import (
    AudioCallback,
    AudioGenRequestBuilder,
    BudgetCheck,
    DirectRequestBuilder,
    ImageCallback,
    ImageRequestBuilder,
    JsonLLMResponse,
    JsonRequestBuilder,
    LLMCallback,
    MediaBuilder,
    OnUsage,
    StreamingToolExecutor,
    ToolExecutor,
    ToolProgress,
    ToolResult,
    ToolStreamItem,
    TTSRequestBuilder,
    TypedLLMResponse,
    UsageCallback,
    VideoCallback,
    VideoRequestBuilder,
    select_best,
)
from .tokens import count, estimate_input_tokens

try:
    __version__ = version("gate-llmax")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "AudioCallback",
    "AudioGenMode",
    "AudioGenRequest",
    "AudioGenRequestBuilder",
    "AudioGenResponse",
    "AudioIsolationRequest",
    "AudioIsolationResponse",
    "AudioMode",
    "AudioRequest",
    "AudioResponse",
    "BaseAudioResponse",
    "BestTarget",
    "BudgetCheck",
    "CallControl",
    "DeploymentInfo",
    "DeploymentStatus",
    "DialogueTurn",
    "DirectRequestBuilder",
    "DubbingRequest",
    "DubbingResponse",
    "EmbedObject",
    "EmbedRequest",
    "EmbedResponse",
    "ExtraAttributeName",
    "FakeClient",
    "FallbackRung",
    "ImageCallback",
    "ImageData",
    "ImageMessage",
    "ImageRequest",
    "ImageRequestBuilder",
    "ImageResponse",
    "JsonDict",
    "JsonLLMResponse",
    "JsonRequestBuilder",
    "JsonValue",
    "LLMAuthError",
    "LLMBudgetError",
    "LLMCallRecord",
    "LLMCallback",
    "LLMCapabilityError",
    "LLMClient",
    "LLMConnectionError",
    "LLMContentFilterError",
    "LLMContextOverflowError",
    "LLMError",
    "LLMEscapeHatchWarning",
    "LLMModelNotFoundError",
    "LLMResponse",
    "LLMServerError",
    "LLMTimeoutError",
    "MediaBuilder",
    "Message",
    "MessageRole",
    "ModelCapabilities",
    "ModelInfo",
    "ModelPlanRow",
    "ModelPurpose",
    "MulticallStreamFrame",
    "OnUsage",
    "OutputStatus",
    "PlanInfo",
    "RateLimit",
    "RawUsage",
    "ReasoningEffort",
    "RequestSpecifics",
    "ResolveRequest",
    "ResolveResponse",
    "ResolvedDeployment",
    "ResponsesRequest",
    "ResponsesResponse",
    "StreamChunk",
    "StreamingToolExecutor",
    "TTSRequest",
    "TTSRequestBuilder",
    "TTSResponse",
    "TextMessage",
    "ToolCall",
    "ToolExecutor",
    "ToolFunction",
    "ToolProgress",
    "ToolResult",
    "ToolStreamItem",
    "TranscriptionSegment",
    "TypedLLMResponse",
    "UsageCallback",
    "VideoCallback",
    "VideoRequest",
    "VideoRequestBuilder",
    "VideoResponse",
    "VisionLLMResponse",
    "VisionLine",
    "VisionOCR",
    "VisionOCRRequest",
    "VisionPoint",
    "VisionWord",
    "ZoneSelection",
    "__version__",
    "count",
    "estimate_input_tokens",
    "select_best",
]
