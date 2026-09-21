"""Public response/stream/iteration types for the LLM API.

These used to live in src/utils/clients.py and have been moved here as part
of the migration toward src/llm/ owning all non-embedding LLM orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from google import genai
    from openai import AsyncOpenAI

    from src.llm.capture import CapturedMessage


T = TypeVar("T")

# OpenAI GPT-5 specific reasoning levels.
ReasoningEffortType = (
    Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
)
VerbosityType = Literal["low", "medium", "high"] | None

# Raw SDK client union used by the provider-selection layer. The SDK types are
# only imported for type checking; at runtime this stays Any so importing this
# module doesn't load any provider SDK.
if TYPE_CHECKING:
    ProviderClient = AsyncAnthropic | AsyncOpenAI | genai.Client
else:
    ProviderClient = Any


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
    # OpenTelemetry-style span-tree correlation.
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    # Monotonic executor-call ordinal WITHIN a span (total ordering of its
    # steps).
    step_seq: int = 0
    # Retry/fallback attempt within an iteration.
    attempt: int = 1
    # Optional peer context (dream agents pass observer/observed; dialectic
    # passes peer_name). Kept here so AgentIterationEvent can populate
    # them without a separate threading path.
    #
    # `observer` is the tool-loop agent's single observer — it reaches
    # AgentIterationEvent and the Langfuse tags, which only fire for dialectic
    # and the dream specialists, and those have exactly one by construction.
    # `observers` is every collection the call writes to, so the deriver's
    # fan-out has somewhere to go; single-observer agents set both. Set
    # `observers` always; set `observer` only when there is genuinely one.
    observer: str | None = None
    observed: str | None = None
    peer_name: str | None = None
    observers: list[str] = field(default_factory=list)
    # Source work, distinct from messages retrieved later by agent tools.
    source_message_ids: list[str] = field(default_factory=list)
    queue_item_ids: list[int] = field(default_factory=list)
    parent_event_id: str | None = None
    # Used to group traces (should not use session_name because it is not unique)
    session_id: str | None = None
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
    # Per-span memo for O(N) message capture in CapturedLLMCall
    hash_memo: dict[int, CapturedMessage] | None = field(
        default=None, compare=False, repr=False
    )

    def span_identity(self) -> str | None:
        """Effective span id: the new `span_id`, falling back to legacy `run_id`."""
        return self.span_id or self.run_id

    def exported_parent_span_id(self) -> str | None:
        """`parent_span_id` for EXPORT, collapsing the self-parent sentinel to None."""
        pid = self.parent_span_id
        return None if pid is not None and pid == self.span_id else pid


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
            # Closing the public iterator must finalize the provider's trace now,
            # even when the caller stops before the next chunk arrives.
            stream = self._stream
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()
            text = "".join(accumulated_text)
            # Close the run span once, stamping the streamed text as its
            # output. In `finally` so an early-exit caller still closes
            # the span rather than leaking it.
            handle = self._langfuse_run_handle
            if handle is not None:
                self._langfuse_run_handle = None
                handle.end(output=text or None)


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
