"""
Agent events for Honcho telemetry.

Agent events track per-iteration metrics and state-changing tool executions
within agentic loops (dream specialists, dialectic agents). These events
correlate with parent events via run_id for detailed analytics.

Events in this module:
- AgentIterationEvent: Per-LLM-call metrics within an agent run
- AgentToolConclusionsCreatedEvent: Conclusions created by agent tool
- AgentToolConclusionsDeletedEvent: Conclusions deleted by agent tool
- AgentToolPeerCardUpdatedEvent: Peer card updated by agent tool
- AgentToolSummaryCreatedEvent: Summary created by agent tool (future)
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class AgentIterationEvent(BaseEvent):
    """Emitted after each LLM call within an agentic loop.

    Tracks per-iteration token usage and tool calls for detailed cost analysis.
    Correlates with parent events (dream.run, dream.specialist, dialectic.completed)
    via run_id.
    """

    _event_type: ClassVar[str] = "agent.iteration"
    _schema_version: ClassVar[int] = 2
    _category: ClassVar[str] = "agent"
    _volume_class: ClassVar[str] = "high_volume"

    # Run identification
    run_id: str = Field(..., description="Nanoid for run correlation")

    # Context
    parent_category: str = Field(
        ..., description="Parent category: 'dream' or 'dialectic'"
    )
    agent_type: str = Field(
        ..., description="Agent type: 'deduction', 'induction', or 'dialectic'"
    )
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context (for dream agents)
    observer: str | None = Field(default=None, description="Observer peer name")
    observed: str | None = Field(default=None, description="Observed peer name")

    # Peer context (for dialectic agent)
    peer_name: str | None = Field(default=None, description="Peer name being queried")

    # Iteration info
    iteration: int = Field(..., description="Iteration number (1-indexed)")

    # What happened in this iteration
    tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool names called in this iteration (can be empty or multiple)",
    )

    # Token usage for this single LLM call
    input_tokens: int = Field(..., description="Input tokens for this iteration")
    output_tokens: int = Field(..., description="Output tokens for this iteration")
    cache_read_tokens: int = Field(
        default=0, description="Tokens read from prompt cache"
    )
    cache_creation_tokens: int = Field(
        default=0, description="Tokens written to prompt cache"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}"


class AgentToolConclusionsCreatedEvent(BaseEvent):
    """Emitted when the create_conclusions tool is executed.

    Tracks conclusion creation with level breakdown for analytics.
    """

    _event_type: ClassVar[str] = "agent.tool.conclusions.created"
    _schema_version: ClassVar[int] = 2
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="Nanoid for run correlation")
    iteration: int = Field(..., description="Iteration number when this occurred")

    # Context
    parent_category: str = Field(
        ..., description="Parent category: 'dream' or 'dialectic'"
    )
    agent_type: str = Field(
        ..., description="Agent type: 'deduction', 'induction', or 'dialectic'"
    )
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # What was created
    conclusion_count: int = Field(..., description="Number of conclusions created")
    levels: list[str] = Field(
        default_factory=list,
        description="Level of each conclusion (e.g., ['explicit', 'deductive', 'deductive'])",
    )

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}:conclusions_created"


class AgentToolConclusionsDeletedEvent(BaseEvent):
    """Emitted when the delete_conclusions tool is executed.

    Tracks conclusion deletion for memory consolidation analytics.
    """

    _event_type: ClassVar[str] = "agent.tool.conclusions.deleted"
    _schema_version: ClassVar[int] = 3
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="Nanoid for run correlation")
    iteration: int = Field(..., description="Iteration number when this occurred")

    # Context
    parent_category: str = Field(..., description="Parent category (typically 'dream')")
    agent_type: str = Field(
        ..., description="Agent type (typically 'deduction' for deletions)"
    )
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # What was deleted
    conclusion_count: int = Field(..., description="Number of conclusions deleted")
    levels: list[str] = Field(
        default_factory=list,
        description="Level of each deleted conclusion (e.g., ['explicit', 'deductive', 'deductive'])",
    )

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}:conclusions_deleted"


class AgentToolPeerCardUpdatedEvent(BaseEvent):
    """Emitted when the update_peer_card tool is executed.

    Tracks peer card updates during dream consolidation.
    """

    _event_type: ClassVar[str] = "agent.tool.peer_card.updated"
    _schema_version: ClassVar[int] = 2
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="Nanoid for run correlation")
    iteration: int = Field(..., description="Iteration number when this occurred")

    # Context
    parent_category: str = Field(..., description="Parent category (typically 'dream')")
    agent_type: str = Field(..., description="Agent type: 'deduction' or 'induction'")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # What was updated
    facts_count: int = Field(..., description="Number of facts in the peer card")

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}:peer_card_updated"


class AgentToolSummaryCreatedEvent(BaseEvent):
    """Emitted when a summary is created.

    Tracks summary creation with full context about what was summarized
    and the resources consumed.
    """

    _event_type: ClassVar[str] = "agent.tool.summary.created"
    _schema_version: ClassVar[int] = 3
    _category: ClassVar[str] = "agent"

    # Run identification.
    run_id: str | None = Field(
        default=None,
        description="Run id for agentic correlation; None when not in a run",
    )
    iteration: int | None = Field(
        default=None,
        description="Iteration within an agentic loop; None when not in one",
    )

    # Context
    parent_category: str = Field(..., description="Parent category")
    agent_type: str = Field(..., description="Agent type")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_name: str = Field(..., description="Session name")

    # Message context - what was summarized
    message_id: str = Field(..., description="Message ID the summary covers up to")
    message_count: int = Field(
        ..., description="Number of messages included in summary"
    )
    message_seq_in_session: int = Field(
        ..., description="Sequence number of the base message in session"
    )

    # Summary details
    summary_type: str = Field(..., description="Summary type: 'short' or 'long'")

    # Token usage
    input_tokens: int = Field(
        ...,
        description=(
            "Provider-side input tokens for the summary LLM call "
            "(equivalent to HonchoLLMCallResponse.input_tokens). "
            "keeps this field unchanged — adding a duplicate "
            "`provider_input_tokens` would only churn analytics queries."
        ),
    )
    output_tokens: int = Field(..., description="Output tokens (summary token count)")

    # ---- Additive fields ----
    # Breakdown of what *went into* the summary prompt. Lets calibration
    # answer "how much of a summary call's cost was the previous-summary
    # rollup vs. the new messages vs. the scaffold/instructions" without
    # re-deriving from the message corpus.
    previous_summary_tokens: int = Field(
        default=0,
        description=(
            "Token count of the previous summary text fed back in as context. "
            "0 when this is the first summary for the session."
        ),
    )
    message_tokens: int = Field(
        default=0,
        description=(
            "Sum of `Message.token_count` across the messages being "
            "summarized (excludes scaffold and previous_summary)."
        ),
    )
    prompt_scaffold_tokens: int = Field(
        default=0,
        description=(
            "Estimated tokens for the static scaffold portion of the prompt "
            "(from estimate_short/long_summary_prompt_tokens)."
        ),
    )

    def get_resource_id(self) -> str:
        """Idempotency key. A summary is unique per (message it covers up to,
        tier)"""
        return f"{self.message_id}:{self.summary_type}:summary_created"


class AgentToolCallCompletedEvent(BaseEvent):
    """generic tool-call event: fires once per tool invocation.

    Complements the four state-changer events (conclusions_created/deleted,
    peer_card_updated, summary_created), which carry semantic information
    about specific tools, with a lightweight per-call telemetry record that
    covers every tool — including read-only tools (`search_memory`,
    `get_recent_history`, etc.) that have no dedicated event today.

    Resource id includes `tool_call_seq` so the model can legitimately call
    the same tool twice in one iteration (it does) without colliding event
    ids — without seq, both calls would deterministically hash to the same
    id and dedupe would drop one.
    """

    _event_type: ClassVar[str] = "agent.tool.call.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"
    _volume_class: ClassVar[str] = "high_volume"

    # Run identification
    run_id: str = Field(..., description="Run id for correlation")
    iteration: int = Field(..., description="Iteration number (1-indexed)")
    tool_call_seq: int = Field(
        ...,
        description="0-indexed position within the iteration's tool batch. Disambiguates two calls to the same tool in one iteration.",
    )
    provider_tool_call_id: str | None = Field(
        default=None,
        description="Provider-supplied tool call id (e.g. Anthropic's toolu_*) when available; lets analytics cross-reference provider logs",
    )

    # Context
    parent_category: str = Field(
        ..., description="Parent category: 'dream' or 'dialectic'"
    )
    agent_type: str = Field(
        ..., description="Agent type: 'deduction', 'induction', or 'dialectic'"
    )
    workspace_name: str = Field(..., description="Workspace name")

    # What ran
    tool_name: str = Field(..., description="Tool name as invoked")
    duration_ms: float = Field(..., description="Wall-clock duration of the handler")
    is_error: bool = Field(default=False, description="True if the handler raised")

    # Result shape
    result_chars: int = Field(
        ..., description="Length of the result string returned to the LLM"
    )
    result_chars_before_truncation: int | None = Field(
        default=None,
        description="Original result size when the handler truncated; None when no truncation occurred. Pair with was_truncated for the delta.",
    )
    result_tokens_estimate: int = Field(
        default=0,
        description="tiktoken-based size proxy for the result string; estimate only",
    )
    was_truncated: bool = Field(
        default=False,
        description="True when the handler clamped the result to fit a size budget",
    )

    # Search-specific fields (None for non-search tools). Populated by search
    # handlers via the ToolResult.metadata bridge.
    query_tokens: int | None = Field(
        default=None, description="tiktoken estimate of the search query text"
    )
    top_k: int | None = Field(
        default=None, description="Caller-supplied top_k for the search"
    )
    results_count: int | None = Field(
        default=None, description="Number of results returned by the search"
    )
    used_embedding: bool | None = Field(
        default=None,
        description="True when the search ran a vector lookup (vs. metadata-only filter)",
    )
    embedding_query_count: int = Field(
        default=0,
        description="Number of embedding API calls the handler made for this invocation",
    )

    def get_resource_id(self) -> str:
        """{run_id}:{iteration}:{tool_call_seq} so duplicate tool calls within
        one iteration produce distinct deterministic ids."""
        return f"{self.run_id}:{self.iteration}:{self.tool_call_seq}"


__all__ = [
    "AgentIterationEvent",
    "AgentToolCallCompletedEvent",
    "AgentToolConclusionsCreatedEvent",
    "AgentToolConclusionsDeletedEvent",
    "AgentToolPeerCardUpdatedEvent",
    "AgentToolSummaryCreatedEvent",
]
