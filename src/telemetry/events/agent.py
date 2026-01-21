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
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")

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
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")
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
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")
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

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}:conclusions_deleted"


class AgentToolPeerCardUpdatedEvent(BaseEvent):
    """Emitted when the update_peer_card tool is executed.

    Tracks peer card updates during dream consolidation.
    """

    _event_type: ClassVar[str] = "agent.tool.peer_card.updated"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"

    # Run identification
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")
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
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "agent"

    # Run identification (may be placeholder if not from an agentic loop)
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")
    iteration: int = Field(..., description="Iteration number when this occurred")

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
        ..., description="Input tokens used for summary generation"
    )
    output_tokens: int = Field(..., description="Output tokens (summary token count)")

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and iteration for uniqueness."""
        return f"{self.run_id}:{self.iteration}:summary_created"


__all__ = [
    "AgentIterationEvent",
    "AgentToolConclusionsCreatedEvent",
    "AgentToolConclusionsDeletedEvent",
    "AgentToolPeerCardUpdatedEvent",
    "AgentToolSummaryCreatedEvent",
]
