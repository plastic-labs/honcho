"""
Dialectic events for Honcho telemetry.

Dialectic tasks answer questions about peers by gathering context from memory.
These are user-initiated operations that query the representation system.
The run_id field enables correlation with agent.iteration events.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DialecticCompletedEvent(BaseEvent):
    """Emitted when a dialectic (chat) query completes.

    Dialectic queries answer questions about peers by gathering context
    from memory. This event captures the full context of the query and
    its execution metrics.

    The run_id correlates with AgentIterationEvent and AgentTool* events
    for detailed analytics.
    """

    _event_type: ClassVar[str] = "dialectic.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dialectic"

    # Run identification (for correlating with iteration/tool events)
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")

    # Workspace context
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    peer_name: str = Field(..., description="Peer name being queried about")

    # Session context (optional - dialectic can run without session)
    session_name: str | None = Field(
        default=None, description="Session name if provided"
    )

    # Query configuration
    reasoning_level: str = Field(
        ..., description="Reasoning level: minimal, low, medium, high, max"
    )

    # Execution metrics
    total_iterations: int = Field(default=1, description="Number of LLM iterations")
    prefetched_conclusion_count: int = Field(
        default=0, description="Number of conclusions prefetched"
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
        """Resource ID is the run_id for uniqueness."""
        return self.run_id


__all__ = ["DialecticCompletedEvent"]
