"""
Concrete CloudEvents event types for Honcho telemetry.

This module defines all the event types emitted by Honcho for analytics
and observability purposes. Events are grouped into two categories:

- Work events: Background processing tasks (representation, summary, dream, etc.)
- Activity events: User-initiated operations (dialectic queries)

Each event captures comprehensive dimensions for analytics, including:
- Resource identifiers (workspace, session, peer IDs)
- Operation metrics (counts, durations, token usage)
- Status information (success/failure states)
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent

# =============================================================================
# Work Events - Background Processing
# =============================================================================


class RepresentationCompletedEvent(BaseEvent):
    """Emitted when a representation task completes processing a message batch.

    Representation tasks extract observations from messages to build peer
    representations. This event captures the full context of what was processed
    and what was extracted.
    """

    _event_type: ClassVar[str] = "honcho.representation.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "work"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_id: str = Field(..., description="Session ID")
    session_name: str = Field(..., description="Session name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Message batch info
    earliest_message_id: str = Field(..., description="First message ID in batch")
    latest_message_id: str = Field(..., description="Last message ID in batch")
    message_count: int = Field(..., description="Number of messages processed")

    # Observation counts
    explicit_observation_count: int = Field(
        ..., description="Number of explicit observations extracted"
    )
    deductive_observation_count: int = Field(
        ..., description="Number of deductive observations inferred"
    )

    # Timing metrics (milliseconds)
    context_preparation_ms: float = Field(
        ..., description="Time spent preparing context"
    )
    llm_call_ms: float = Field(..., description="Time spent in LLM call")
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, session, and latest message for uniqueness."""
        return f"{self.workspace_id}:{self.session_id}:{self.latest_message_id}"


class SummaryCompletedEvent(BaseEvent):
    """Emitted when a session summary is created or updated.

    Summary tasks create rolling summaries of session history at two granularities:
    - Short summaries: Every 20 messages
    - Long summaries: Every 60 messages
    """

    _event_type: ClassVar[str] = "honcho.summary.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "work"

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


class DreamCompletedEvent(BaseEvent):
    """Emitted when a dream (memory consolidation) task completes.

    Dreams run periodically to consolidate observations, identify patterns,
    and improve memory quality through deduction and induction specialists.
    """

    _event_type: ClassVar[str] = "honcho.dream.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "work"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context (most recent session with documents)
    session_name: str = Field(..., description="Most recent session name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Dream details
    dream_type: str = Field(..., description="Dream type (e.g., 'omni')")

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
        return f"{self.workspace_id}:{self.observer}:{self.observed}:{self.dream_type}"


class ReconciliationCompletedEvent(BaseEvent):
    """Emitted when a reconciliation cycle completes.

    Reconciliation tasks sync documents and message embeddings to external
    vector stores and clean up soft-deleted records. These run periodically
    and operate across all workspaces.

    Note: This event has no workspace context as it operates globally.
    """

    _event_type: ClassVar[str] = "honcho.reconciliation.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "work"

    # Reconciler type
    reconciler_type: str = Field(
        ..., description="Reconciler type: 'sync_vectors' or 'cleanup_queue'"
    )

    # Document metrics
    documents_synced: int = Field(
        default=0, description="Documents successfully synced to vector store"
    )
    documents_failed: int = Field(
        default=0, description="Documents that failed to sync"
    )
    documents_cleaned: int = Field(
        default=0, description="Soft-deleted documents cleaned up"
    )

    # Message embedding metrics
    message_embeddings_synced: int = Field(
        default=0, description="Message embeddings successfully synced"
    )
    message_embeddings_failed: int = Field(
        default=0, description="Message embeddings that failed to sync"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID is just the reconciler type since it's a global operation."""
        return self.reconciler_type


class DeletionCompletedEvent(BaseEvent):
    """Emitted when a deletion task completes.

    Deletion tasks handle async removal of sessions and observations,
    ensuring proper cleanup across all storage layers.
    """

    _event_type: ClassVar[str] = "honcho.deletion.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "work"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Deletion details
    deletion_type: str = Field(
        ..., description="Type of deletion: 'session' or 'observation'"
    )
    resource_id: str = Field(..., description="ID of the deleted resource")

    # Outcome
    success: bool = Field(..., description="Whether deletion succeeded")

    # Optional error info
    error_message: str | None = Field(
        default=None, description="Error message if deletion failed"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, type, and resource for uniqueness."""
        return f"{self.workspace_id}:{self.deletion_type}:{self.resource_id}"


# =============================================================================
# Activity Events - User-Initiated Operations
# =============================================================================


class DialecticCompletedEvent(BaseEvent):
    """Emitted when a dialectic (chat) query completes.

    Dialectic queries answer questions about peers by gathering context
    from memory. This event captures the full context of the query and
    its execution metrics.
    """

    _event_type: ClassVar[str] = "honcho.dialectic.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "activity"

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    peer_id: str = Field(..., description="Peer ID being queried about")
    peer_name: str = Field(..., description="Peer name being queried about")

    # Session context (optional - dialectic can run without session)
    session_id: str | None = Field(default=None, description="Session ID if provided")
    session_name: str | None = Field(
        default=None, description="Session name if provided"
    )

    # Query configuration
    reasoning_level: str = Field(
        ..., description="Reasoning level: minimal, low, medium, high, max"
    )
    agentic: bool = Field(..., description="Whether agentic mode was used")

    # Execution metrics
    prefetched_observation_count: int = Field(
        default=0, description="Number of observations prefetched"
    )
    tool_calls_count: int = Field(default=0, description="Number of tool calls made")

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage with cache breakdown
    input_tokens: int = Field(..., description="Total input tokens")
    output_tokens: int = Field(..., description="Output tokens generated")
    cache_read_tokens: int = Field(
        default=0, description="Tokens read from prompt cache"
    )
    cache_creation_tokens: int = Field(
        default=0, description="Tokens written to prompt cache"
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, peer, and session for uniqueness."""
        session_part = self.session_id or "none"
        return f"{self.workspace_id}:{self.peer_id}:{session_part}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Work events
    "RepresentationCompletedEvent",
    "SummaryCompletedEvent",
    "DreamCompletedEvent",
    "ReconciliationCompletedEvent",
    "DeletionCompletedEvent",
    # Activity events
    "DialecticCompletedEvent",
]
