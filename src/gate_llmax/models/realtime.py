"""Realtime (live voice) session config and the unified event wire shared by client and backend."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Any

from pydantic import BaseModel, Field

from .request import CallControl
from .response import LLMCallRecord


class RealtimeSessionConfig(CallControl):
    """The provider-agnostic slice of a realtime session's open parameters; vendor knobs live on the deployment."""

    model: str
    system_instruction: str | None = None
    tools: list[dict[str, Any]] | None = None
    voice: str | None = None
    language: str | None = None
    output_sample_rate: int = 48000
    enable_input_audio_transcription: bool = True
    enable_output_audio_transcription: bool = True
    silence_duration_ms: int | None = None
    initial_message: str | None = None


@unique
class RealtimeEventType(StrEnum):
    """The unified realtime event kinds every provider is translated into."""

    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    INPUT_TRANSCRIPT = "input_transcript"
    OUTPUT_TRANSCRIPT = "output_transcript"
    TOOL_CALL = "tool_call"
    TOOL_CALL_CANCELLATION = "tool_call_cancellation"
    ERROR = "error"
    GENERATION_COMPLETE = "generation_complete"
    TURN_COMPLETE = "turn_complete"
    INTERRUPTED = "interrupted"
    VOICE_ACTIVITY = "voice_activity"
    GO_AWAY = "go_away"
    USAGE_METADATA = "usage_metadata"


class RealtimeAudioData(BaseModel):
    """A chunk of model audio; ``data`` is base64 PCM16 on the wire."""

    data: str
    sample_rate: int
    item_id: str = ""


class RealtimeTranscriptData(BaseModel):
    """Incremental model output text."""

    text: str
    item_id: str = ""
    is_final: bool = False


class RealtimeInputTranscriptData(BaseModel):
    """Server transcription of the user's audio."""

    text: str


class RealtimeOutputTranscriptData(BaseModel):
    """Authoritative server transcription of the model's audio."""

    text: str
    item_id: str = ""
    is_final: bool = False


class RealtimeToolCallData(BaseModel):
    """A tool call the model requested; ``arguments`` is a JSON string."""

    tool_call_id: str
    name: str
    arguments: str


class RealtimeToolCallCancellationData(BaseModel):
    """Ids of tool calls the model cancelled."""

    ids: list[str]


class RealtimeErrorData(BaseModel):
    """An error surfaced from the session."""

    message: str
    code: str | None = None
    recoverable: bool = True


class RealtimeInterruptedData(BaseModel):
    """Barge-in; the model's current item was cut off."""

    item_id: str | None = None


class RealtimeVoiceActivityData(BaseModel):
    """VAD edge; ``activity_type`` is "start" or "end"."""

    activity_type: str


class RealtimeGoAwayData(BaseModel):
    """Session migration/expiry warning."""

    time_left: str | None = None


class RealtimeUsageMetadataData(BaseModel):
    """Modality-split token counts for billing; ``input_cached_tokens`` is a subset of the input counts."""

    total_token_count: int = 0
    input_audio_tokens: int = 0
    input_text_tokens: int = 0
    input_cached_tokens: int = 0
    output_audio_tokens: int = 0
    output_text_tokens: int = 0

    def __add__(self, other: RealtimeUsageMetadataData) -> RealtimeUsageMetadataData:
        """Accumulate two usage snapshots (realtime bills per session, not per turn)."""
        return RealtimeUsageMetadataData(
            total_token_count=self.total_token_count + other.total_token_count,
            input_audio_tokens=self.input_audio_tokens + other.input_audio_tokens,
            input_text_tokens=self.input_text_tokens + other.input_text_tokens,
            input_cached_tokens=self.input_cached_tokens + other.input_cached_tokens,
            output_audio_tokens=self.output_audio_tokens + other.output_audio_tokens,
            output_text_tokens=self.output_text_tokens + other.output_text_tokens,
        )


class RealtimeEvent(BaseModel):
    """One unified realtime frame: a type plus the single payload that type carries."""

    type: RealtimeEventType
    audio: RealtimeAudioData | None = None
    transcript: RealtimeTranscriptData | None = None
    input_transcript: RealtimeInputTranscriptData | None = None
    output_transcript: RealtimeOutputTranscriptData | None = None
    tool_call: RealtimeToolCallData | None = None
    tool_call_cancellation: RealtimeToolCallCancellationData | None = None
    error: RealtimeErrorData | None = None
    interrupted: RealtimeInterruptedData | None = None
    voice_activity: RealtimeVoiceActivityData | None = None
    go_away: RealtimeGoAwayData | None = None
    usage_metadata: RealtimeUsageMetadataData | None = None


class RealtimeRecord(LLMCallRecord):
    """The final per-session billed record, logged once at teardown; carries the aggregated token usage."""

    usage_metadata: RealtimeUsageMetadataData = Field(default_factory=RealtimeUsageMetadataData)
    session_seconds: float = 0.0
