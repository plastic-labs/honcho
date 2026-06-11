"""LLM-call events for Honcho telemetry.

These events fire once per provider hit (each iteration of an agentic tool loop,
each deriver/summarizer LLM call, etc.) and carry the full cost-attribution
context: model/provider/transport, token counts with cache breakdown, finish
reason, outcome (success/error), retry/fallback state, and run correlation.

Unlike the existing aggregate `*Completed` events (representation, dialectic,
dream), this event is high-volume. It participates in the
`settings.TELEMETRY.HIGH_VOLUME_SAMPLE_RATE` sampler so per-iteration emission
can be tuned against a budget.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Literal

from pydantic import Field

from src.config import ModelTransport
from src.telemetry.events.base import BaseEvent


class CallPurpose(str, Enum):
    """Closed taxonomy for LLM call purposes.

    The schema lint enforces that all `LLMCallCompletedEvent` emissions use a
    value from this enum. Adding a new call site requires adding a value here
    first — keeps the analytics taxonomy stable.
    """

    DERIVER_REPRESENTATION = "deriver.representation"
    DIALECTIC_ANSWER = "dialectic.answer"
    DREAM_DEDUCTION = "dream.deduction"
    DREAM_INDUCTION = "dream.induction"
    SUMMARY_SHORT = "summary.short"
    SUMMARY_LONG = "summary.long"


class LLMCallCompletedEvent(BaseEvent):
    """Emitted once per provider hit by `honcho_llm_call_inner`.

    Covers success, failure, and cancellation via `outcome`. The last attempt
    of a tenacity retry chain is flagged with `is_final_attempt=True` regardless
    of outcome — calibration queries for "exhausted" use
    `outcome='error' AND is_final_attempt`. Cancellations (typically client
    disconnect mid-stream or server shutdown) are distinct from errors and
    should not feed error-rate alerting.

    Streaming note: when `was_stream=True`, the token counts are placeholders
    (0) because token totals aren't knowable until the stream drains. Use the
    aggregate envelopes (`DialecticCompletedEvent` etc.) for streamed-call
    accuracy until streaming completion is wired through.
    """

    _event_type: ClassVar[str] = "llm.call.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "llm"
    _volume_class: ClassVar[str] = "high_volume"

    # Context (None for system calls without workspace context)
    workspace_name: str | None = Field(default=None, description="Workspace name")
    call_purpose: CallPurpose | None = Field(
        default=None,
        description="Closed enum identifying the call site (deriver, dialectic, etc.)",
    )
    parent_category: str | None = Field(
        default=None,
        description="Parent category for analytics joins: 'representation' | 'dialectic' | 'dream' | 'summary'",
    )

    # Provider info
    transport: ModelTransport = Field(
        ..., description="SDK transport: 'anthropic' | 'openai' | 'gemini'"
    )
    provider_label: str | None = Field(
        default=None,
        description="Best-effort vendor inference for relay setups (e.g. 'anthropic' when an OpenRouter base_url + 'anthropic/claude-...' model is used); None when not reliably inferable",
    )
    model: str = Field(..., description="Model identifier as sent to the provider")
    effective_max_output_tokens: int = Field(
        ..., description="max_tokens value used for this call"
    )

    # Token usage (zero on was_stream=True placeholder)
    provider_input_tokens: int = Field(default=0, description="Provider input_tokens")
    provider_output_tokens: int = Field(default=0, description="Provider output_tokens")
    cache_read_tokens: int = Field(
        default=0, description="Tokens read from prompt cache"
    )
    cache_creation_tokens: int = Field(
        default=0, description="Tokens written to prompt cache"
    )

    # Outcome
    finish_reason: str | None = Field(
        default=None,
        description="First finish reason from the response (None on error)",
    )
    outcome: Literal["success", "error", "cancelled"] = Field(
        ...,
        description="'success' when the provider returned a result, 'error' when it raised, 'cancelled' when the awaitable was cancelled (client disconnect, server shutdown). Cancellations should be excluded from error-rate alerting.",
    )
    is_final_attempt: bool = Field(
        ...,
        description="True when this is the last allowed attempt (attempt == retry_attempts). Combine with outcome='error' to identify retry-exhausted calls. Cancellations are not retried so this reflects the attempt at cancellation time.",
    )
    error_class: str | None = Field(
        default=None,
        description="Exception class name when outcome is 'error' or 'cancelled' (e.g. 'CancelledError')",
    )

    # Retry/fallback state
    attempt: int = Field(..., description="1-indexed tenacity attempt number")
    retry_attempts: int = Field(..., description="Total attempts allowed by caller")
    was_fallback: bool = Field(
        ..., description="True when this attempt used the fallback ModelConfig"
    )

    # Timing
    duration_ms: float = Field(
        ..., description="Wall-clock duration of the provider call"
    )

    # Shape
    has_tools: bool = Field(default=False, description="True if tools were provided")
    tool_call_count: int = Field(
        default=0, description="Number of tool calls the model requested"
    )
    was_stream: bool = Field(
        default=False,
        description="True for the stream_final_response path. Token counts are 0 placeholders — see class docstring.",
    )

    # Agent correlation (None for non-agent calls like summarizer / deriver)
    run_id: str | None = Field(
        default=None,
        description="Agent run id (ULID when widened in follow-up)",
    )
    iteration: int | None = Field(
        default=None,
        description="1-indexed iteration within an agentic tool loop. Passed explicitly via LLMTelemetryContext — NOT read from set_current_iteration (that fires after the LLM call)",
    )

    def get_resource_id(self) -> str:
        """Resource id includes run_id + iteration + attempt + transport/model
        so multi-attempt retries within one iteration get distinct ids."""
        run = self.run_id or "none"
        iteration = self.iteration if self.iteration is not None else 0
        return f"{run}:{iteration}:{self.attempt}:{self.transport}:{self.model}"


class EmbeddingCallPurpose(str, Enum):
    """Closed taxonomy for embedding call purposes.

    Mirrors `CallPurpose` for LLM calls. Adding a new embedding call site
    requires adding a value here first — keeps the analytics taxonomy stable
    and prevents free-form `track_name` drift from leaking into queries.
    """

    SEARCH_MEMORY = "search_memory"
    SEARCH_MESSAGES = "search_messages"
    CREATE_OBSERVATIONS = "create_observations"
    VECTOR_SYNC = "vector_sync"
    SUMMARY = "summary"
    # Pending MessageEmbedding rows from create_messages; embedding runs in the
    # reconciler (not inline on the API path). Distinct from VECTOR_SYNC, which
    # covers document re-embeds and other vector-store healing work.
    MESSAGE_CREATE = "message_create"
    # Added so previously-unattributed call sites land on a distinct slug
    # instead of None. Closed taxonomy — coordinate with analytics before
    # adding more.
    DIALECTIC_PREFETCH = "dialectic_prefetch"
    SESSION_CONTEXT_SEARCH = "session_context_search"
    PREFERENCE_EXTRACTION = "preference_extraction"
    GENERIC_DOCUMENT_SEARCH = "generic_document_search"


class EmbeddingCallCompletedEvent(BaseEvent):
    """Emitted once per embedding-provider call.

    Embedding calls are real provider spend (per-token like LLM calls).
    Search tools, observation creation, the message-embedding sync, and
    the deriver/summarizer paths all hit the embedding API; this event
    captures cost-attribution context for all of them.

    Volume note: this event is high-volume. Interactive paths
    (`search_memory` / `search_messages`) emit one event per query, so under
    a search-heavy dialectic load this can match or exceed the LLM call
    rate. The shared `HIGH_VOLUME_SAMPLE_RATE` covers both.
    """

    _event_type: ClassVar[str] = "embedding.call.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "llm"
    _volume_class: ClassVar[str] = "high_volume"

    workspace_name: str | None = Field(default=None, description="Workspace name")
    call_purpose: EmbeddingCallPurpose | None = Field(
        default=None,
        description=(
            "Closed enum identifying the call site. Set by callers via the "
            "`embedding_call_purpose` ContextVar; None when the call originated "
            "outside an instrumented path."
        ),
    )
    parent_category: str | None = Field(
        default=None,
        description="Parent category for analytics joins (e.g. 'dialectic', 'representation')",
    )

    provider: str = Field(..., description="'openai' | 'gemini'")
    model: str = Field(..., description="Model identifier")
    input_count: int = Field(
        ..., description="Number of texts embedded in this call (batch size)"
    )
    input_tokens_estimate: int = Field(
        default=0,
        description=(
            "tiktoken-based size proxy for the embedded text. ESTIMATE only — "
            "the embedding client uses encoding_for_model() with a cl100k_base "
            "fallback (see embedding_client.py:68-71), which is exact for "
            "older OpenAI models, an approximation for newer ones, and a "
            "rough proxy for Gemini (which has its own tokenizer)."
        ),
    )
    duration_ms: float = Field(
        ..., description="Wall-clock duration of the provider call"
    )

    outcome: Literal["success", "error", "cancelled"] = Field(
        ...,
        description="'success' when the provider returned a result, 'error' when it raised, 'cancelled' when the awaitable was cancelled. Cancellations should be excluded from error-rate alerting.",
    )
    is_final_attempt: bool = Field(
        default=False,
        description=(
            "True on the last retry attempt. Mirrors LLMCallCompletedEvent's "
            "convention: combine with outcome='error' to identify exhausted "
            "embedding calls. Cancellations are not retried."
        ),
    )
    error_class: str | None = Field(
        default=None,
        description="Exception class name when outcome is 'error' or 'cancelled'",
    )

    run_id: str | None = Field(
        default=None,
        description="Agent run id when called from an agentic loop; None for sync/CRUD paths",
    )

    def get_resource_id(self) -> str:
        """Resource id includes timestamp-derived components implicitly via
        generate_id(); we just stake out a non-empty identifier scope."""
        run = self.run_id or "none"
        purpose = self.call_purpose.value if self.call_purpose else "unknown"
        return f"{run}:{purpose}:{self.provider}:{self.model}:{self.input_count}"


__all__ = [
    "CallPurpose",
    "EmbeddingCallCompletedEvent",
    "EmbeddingCallPurpose",
    "LLMCallCompletedEvent",
]
