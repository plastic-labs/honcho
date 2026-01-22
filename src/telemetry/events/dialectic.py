"""
Dialectic events for Honcho telemetry.

Dialectic tasks answer questions about peers by gathering context from memory.
These are user-initiated operations that query the representation system.
The run_id field enables correlation with agent.iteration events.
"""

from typing import ClassVar

from pydantic import BaseModel, Field

from src.telemetry.events.base import BaseEvent


class DialecticPhaseMetrics(BaseModel):
    """Metrics for a single phase of dialectic execution.

    In two-phase mode, there are separate metrics for search and synthesis.
    In single-model mode, there is one phase with combined metrics.
    """

    phase_name: str = Field(
        ..., description="Phase identifier: 'single', 'search', or 'synthesis'"
    )
    provider: str | None = Field(default=None, description="LLM provider used")
    model: str | None = Field(default=None, description="Model name used")
    input_tokens: int = Field(default=0, description="Input tokens for this phase")
    output_tokens: int = Field(default=0, description="Output tokens for this phase")
    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")
    cache_creation_tokens: int = Field(default=0, description="Tokens written to cache")
    iterations: int = Field(default=1, description="LLM call iterations in this phase")
    tool_calls_count: int = Field(
        default=0, description="Tool calls made in this phase"
    )


class DialecticCompletedEvent(BaseEvent):
    """Emitted when a dialectic (chat) query completes.

    Dialectic queries answer questions about peers by gathering context
    from memory. This event captures the full context of the query and
    its execution metrics.

    Supports both single-model and two-phase (search + synthesis) modes.
    In two-phase mode, the `phases` list contains separate metrics for each phase.
    In single-model mode, `phases` contains one entry with combined metrics.

    The run_id correlates with AgentIterationEvent and AgentTool* events
    for detailed analytics.
    """

    _event_type: ClassVar[str] = "dialectic.completed"
    _schema_version: ClassVar[int] = 2  # Bumped for phase metrics support
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

    # Execution mode
    two_phase_mode: bool = Field(
        default=False,
        description="Whether two-phase (search + synthesis) mode was used",
    )

    # Execution metrics (aggregated totals for backward compatibility)
    total_iterations: int = Field(
        default=1, description="Total LLM iterations across all phases"
    )
    prefetched_conclusion_count: int = Field(
        default=0, description="Number of conclusions prefetched"
    )
    tool_calls_count: int = Field(
        default=0, description="Total tool calls across all phases"
    )

    # Timing metrics (milliseconds)
    total_duration_ms: float = Field(..., description="Total processing time")

    # Token usage with cache breakdown (aggregated totals)
    input_tokens: int = Field(..., description="Total input tokens across all phases")
    output_tokens: int = Field(..., description="Total output tokens across all phases")
    cache_read_tokens: int = Field(
        default=0, description="Total tokens read from prompt cache"
    )
    cache_creation_tokens: int = Field(
        default=0, description="Total tokens written to prompt cache"
    )

    # Per-phase metrics (optional, for detailed cost analysis)
    phases: list[DialecticPhaseMetrics] = Field(
        default_factory=list,
        description="Per-phase metrics. Empty for single-model mode, [search, synthesis] for two-phase",
    )

    def get_resource_id(self) -> str:
        """Resource ID is the run_id for uniqueness."""
        return self.run_id


__all__ = ["DialecticCompletedEvent", "DialecticPhaseMetrics"]
