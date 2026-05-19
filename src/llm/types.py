"""Public response/stream/iteration types for the LLM API.

These used to live in src/utils/clients.py and have been moved here as part
of the migration toward src/llm/ owning all non-embedding LLM orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

T = TypeVar("T")

# OpenAI GPT-5 specific reasoning levels.
ReasoningEffortType = (
    Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
)
VerbosityType = Literal["low", "medium", "high"] | None

# Raw SDK client union used by the provider-selection layer.
ProviderClient = AsyncAnthropic | AsyncOpenAI | genai.Client


@dataclass
class IterationData:
    """Data passed to iteration callbacks after each tool execution loop iteration."""

    iteration: int
    """1-indexed iteration number."""
    tool_calls: list[str]
    """List of tool names called in this iteration."""
    input_tokens: int
    """Input tokens used in this iteration's LLM call."""
    output_tokens: int
    """Output tokens generated in this iteration's LLM call."""
    cache_read_tokens: int = 0
    """Tokens read from cache in this iteration."""
    cache_creation_tokens: int = 0
    """Tokens written to cache in this iteration."""


@dataclass
class LLMTelemetryContext:
    """Context threaded through honcho_llm_call → honcho_llm_call_inner so the
    LLMCallCompletedEvent emitter (and AgentIterationEvent emitter)
    can attribute calls to the right workspace / agent / iteration without
    re-deriving any of it from ambient state.

    Iteration is mutable: tool_loop updates this field before each inner call.
    NOT read from set_current_iteration ContextVar — that fires after the LLM
    call returns, so reading it from the executor would yield stale values.
    """

    workspace_name: str | None = None
    # call_purpose carries the same string as src.telemetry.events.llm.CallPurpose values.
    # Stored as str rather than importing the enum here to keep src/llm/ free of
    # telemetry imports — the emitter validates against the enum.
    call_purpose: str | None = None
    parent_category: str | None = None
    run_id: str | None = None
    iteration: int | None = None
    # Optional peer context (dream agents pass observer/observed; dialectic
    # passes peer_name). Kept here so AgentIterationEvent can populate
    # them without a separate threading path.
    observer: str | None = None
    observed: str | None = None
    peer_name: str | None = None
    # Tool-related context: agent_type is the human-readable identifier of the
    # agent — dialectic/deduction/induction. Used by agent iteration
    # event and tool call event.
    agent_type: str | None = None


IterationCallback = Callable[[IterationData], None]


class HonchoLLMCallResponse(BaseModel, Generic[T]):
    """Response object for LLM calls.

    Note:
        Uncached input tokens = input_tokens - cache_read_input_tokens
                              + cache_creation_input_tokens
        (cache_creation costs 25% more, cache_read costs 90% less)
    """

    content: T
    input_tokens: int = 0
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    finish_reasons: list[str]
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    """Number of LLM calls made in the tool execution loop."""
    thinking_content: str | None = None
    # Full thinking blocks with signatures for multi-turn replay (Anthropic only).
    thinking_blocks: list[dict[str, Any]] = Field(default_factory=list)
    # OpenRouter reasoning_details for Gemini models — must be preserved across turns.
    reasoning_details: list[dict[str, Any]] = Field(default_factory=list)
    # True when honcho_llm_call truncated the input messages to fit
    # `max_input_tokens` before dispatching to the provider. Lets the deriver
    # populate `hit_input_token_cap` on RepresentationCompletedEvent with a
    # real measurement (not just the configured cap).
    input_was_truncated: bool = False


class HonchoLLMCallStreamChunk(BaseModel):
    """A single chunk in a streaming LLM response."""

    content: str
    is_done: bool = False
    finish_reasons: list[str] = Field(default_factory=list)
    output_tokens: int | None = None


class StreamingResponseWithMetadata:
    """Streaming response wrapper carrying metadata from a completed tool loop.

    Lets callers read tool_calls_made / token counts / thinking_content from
    the tool-execution phase while still iterating the final streamed answer.
    """

    _stream: AsyncIterator[HonchoLLMCallStreamChunk]
    tool_calls_made: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    thinking_content: str | None
    iterations: int
    input_was_truncated: bool

    def __init__(
        self,
        stream: AsyncIterator[HonchoLLMCallStreamChunk],
        tool_calls_made: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        thinking_content: str | None = None,
        iterations: int = 0,
        input_was_truncated: bool = False,
    ):
        self._stream = stream
        self.tool_calls_made = tool_calls_made
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.thinking_content = thinking_content
        self.iterations = iterations
        self.input_was_truncated = input_was_truncated

    def __aiter__(self) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        return self._stream.__aiter__()

    async def __anext__(self) -> HonchoLLMCallStreamChunk:
        return await self._stream.__anext__()


__all__ = [
    "HonchoLLMCallResponse",
    "HonchoLLMCallStreamChunk",
    "IterationCallback",
    "IterationData",
    "LLMTelemetryContext",
    "ProviderClient",
    "ReasoningEffortType",
    "StreamingResponseWithMetadata",
    "T",
    "VerbosityType",
]
