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
    # Human-readable name for the Langfuse trace + per-call generation
    # (e.g. "Dialectic Agent", "Minimal Deriver"). Sole home for this name —
    # callers set it here; `honcho_llm_call` no longer takes a separate kwarg.
    # Also used to label the sentry `ai_track` decorator and as the source for
    # the run-level `langfuse_agent_run` label.
    track_name: str | None = None


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
    # True when the original input exceeded `max_input_tokens` — covers
    # both "messages were dropped" and "couldn't drop the last unit and
    # remaining tokens still exceeded the cap" (the deriver's prompt-only
    # case). Maps 1:1 to `RepresentationCompletedEvent.hit_input_token_cap`
    # and `DialecticCompletedEvent.hit_input_token_cap`.
    hit_input_token_cap: bool = False


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

    `output_tokens` is updated AS THE STREAM DRAINS — `__aiter__` wraps the
    underlying iterator and accumulates the latest non-None `output_tokens`
    value reported by chunk usage. Providers like OpenAI (with
    `stream_options.include_usage`) and Anthropic emit a final usage chunk
    with the cumulative count, so the post-drain `output_tokens` value
    reflects tool-loop output + final-stream output. Callers that read
    `output_tokens` AFTER fully iterating the stream get the true total;
    callers that read it before drain see only the tool-loop portion.

    `langfuse_run_handle` (optional) is the run-level Langfuse span handle
    transferred from `honcho_llm_call` when streaming. The wrapper owns it
    after construction: on drain, the accumulated streamed text is stamped
    as the run span's output and the span is closed. Without this transfer,
    streaming traces would show blank output because the synchronous return
    happens before any chunks arrive.
    """

    _stream: AsyncIterator[HonchoLLMCallStreamChunk]
    tool_calls_made: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    thinking_content: str | None
    iterations: int
    hit_input_token_cap: bool
    _langfuse_run_handle: Any | None

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
        hit_input_token_cap: bool = False,
        langfuse_run_handle: Any | None = None,
    ):
        self._stream = stream
        self.tool_calls_made = tool_calls_made
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.thinking_content = thinking_content
        self.iterations = iterations
        self.hit_input_token_cap = hit_input_token_cap
        self._langfuse_run_handle = langfuse_run_handle

    def __aiter__(self) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        # Wrap the underlying iterator to capture final-stream output_tokens
        # from chunks as they arrive. Providers emit a usage chunk at end-of-
        # stream with the cumulative output_tokens count; we fold it into
        # self.output_tokens (which carries the tool-loop running total at
        # construction) so the post-drain value reflects the true cost.
        return self._iterate_with_usage_capture()

    async def _iterate_with_usage_capture(
        self,
    ) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        final_stream_output_tokens = 0
        # Only accumulate when a Langfuse run handle is attached — for non-
        # traced streams the buffer is dead weight.
        accumulate = self._langfuse_run_handle is not None
        accumulated_text: list[str] = []
        try:
            async for chunk in self._stream:
                if chunk.output_tokens is not None:
                    # Take the LATEST value, not the sum — providers report
                    # the cumulative usage in the final chunk, not deltas.
                    final_stream_output_tokens = chunk.output_tokens
                if accumulate and chunk.content:
                    accumulated_text.append(chunk.content)
                yield chunk
            # Stream drained — fold the final-stream output tokens into the
            # tool-loop totals so DialecticCompletedEvent / downstream readers
            # see the true cost.
            if final_stream_output_tokens > 0:
                self.output_tokens += final_stream_output_tokens
        finally:
            # Close the run span once, stamping the streamed text as its
            # output. In `finally` so an early-exit caller still closes
            # the span rather than leaking it.
            handle = self._langfuse_run_handle
            if handle is not None:
                self._langfuse_run_handle = None
                handle.end(output="".join(accumulated_text) or None)


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
