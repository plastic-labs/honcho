"""
Dream events for Honcho telemetry.

Dream tasks handle memory consolidation operations including:
- Session summarization
- Peer card updates
- Deductive reasoning (inferring new facts from existing observations)
- Inductive reasoning (generalizing patterns from observations)
- Omni dreams (comprehensive consolidation combining multiple operations)
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DreamSummaryCompletedEvent(BaseEvent):
    """Emitted when a session summary is created or updated.

    Summary tasks create rolling summaries of session history at two granularities:
    - Short summaries: Every 20 messages
    - Long summaries: Every 60 messages
    """

    _event_type: ClassVar[str] = "dream.summary.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_id: str = Field(..., description="Session ID")
    session_name: str = Field(..., description="Session name")

    # Message context
    message_id: str = Field(..., description="Trigger message ID")
    message_seq_in_session: int = Field(
        ..., description="Message sequence number in session"
    )

    # Summary details
    summary_type: str = Field(..., description="Summary type: 'short' or 'long'")
    summary_token_count: int = Field(
        ..., description="Token count of generated summary"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, session, message, and type for uniqueness."""
        return f"{self.workspace_id}:{self.session_id}:{self.message_id}:{self.summary_type}"


class DreamPeerCardCompletedEvent(BaseEvent):
    """Emitted when a peer card update completes.

    Peer card updates consolidate observations into a structured summary
    of what is known about a peer.
    """

    _event_type: ClassVar[str] = "dream.peer_card.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Update details
    card_updated: bool = Field(..., description="Whether the peer card was updated")

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace and peer context for uniqueness."""
        return f"{self.workspace_id}:{self.observer}:{self.observed}:peer_card"


class DreamDeductiveCompletedEvent(BaseEvent):
    """Emitted when a deductive reasoning task completes.

    Deductive tasks infer new facts from existing observations through
    logical reasoning and pattern matching.
    """

    _event_type: ClassVar[str] = "dream.deductive.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Deduction results
    observations_created: int = Field(
        default=0, description="Number of new observations created"
    )
    observations_deleted: int = Field(
        default=0, description="Number of observations deleted/consolidated"
    )
    success: bool = Field(..., description="Whether deduction succeeded")

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace and peer context for uniqueness."""
        return f"{self.workspace_id}:{self.observer}:{self.observed}:deductive"


class DreamInductiveCompletedEvent(BaseEvent):
    """Emitted when an inductive reasoning task completes.

    Inductive tasks generalize patterns from existing observations,
    creating higher-level insights about the peer.
    """

    _event_type: ClassVar[str] = "dream.inductive.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Induction results
    observations_created: int = Field(
        default=0, description="Number of new observations created"
    )
    observations_deleted: int = Field(
        default=0, description="Number of observations deleted/consolidated"
    )
    success: bool = Field(..., description="Whether induction succeeded")

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace and peer context for uniqueness."""
        return f"{self.workspace_id}:{self.observer}:{self.observed}:inductive"


class DreamOmniCompletedEvent(BaseEvent):
    """Emitted when an omni dream task completes.

    Omni dreams run comprehensive memory consolidation that may include
    multiple specialist operations (deduction, induction, peer card updates).
    This event provides an aggregate view of the entire dream cycle.
    """

    _event_type: ClassVar[str] = "dream.omni.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context (most recent session with documents)
    session_name: str = Field(..., description="Most recent session name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Surprisal sampling (optional - may not run if disabled)
    surprisal_observation_count: int = Field(
        default=0, description="High-surprisal observations found"
    )

    # Specialist outcomes
    deduction_success: bool = Field(
        ..., description="Whether deduction specialist succeeded"
    )
    induction_success: bool = Field(
        ..., description="Whether induction specialist succeeded"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Aggregated token usage from specialists
    input_tokens: int = Field(..., description="Total input tokens used")
    output_tokens: int = Field(..., description="Total output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, peers, and dream type for uniqueness."""
        return f"{self.workspace_id}:{self.observer}:{self.observed}:omni"


__all__ = [
    "DreamSummaryCompletedEvent",
    "DreamPeerCardCompletedEvent",
    "DreamDeductiveCompletedEvent",
    "DreamInductiveCompletedEvent",
    "DreamOmniCompletedEvent",
]
