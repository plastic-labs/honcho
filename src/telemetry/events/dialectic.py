"""
Dialectic events for Honcho telemetry.

Dialectic tasks answer questions about peers by gathering context from memory.
These are user-initiated operations that query the representation system.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DialecticCompletedEvent(BaseEvent):
    """Emitted when a dialectic (chat) query completes.

    Dialectic queries answer questions about peers by gathering context
    from memory. This event captures the full context of the query and
    its execution metrics.
    """

    _event_type: ClassVar[str] = "dialectic.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dialectic"

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


__all__ = ["DialecticCompletedEvent"]
