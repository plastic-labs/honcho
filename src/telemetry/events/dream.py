"""
Dream events for Honcho telemetry.

Dream tasks handle memory consolidation through specialist agents (deduction, induction).
Events track both the overall dream run and individual specialist executions,
with run_id correlation for analytics queries.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class DreamRunEvent(BaseEvent):
    """Emitted when a full dream orchestration completes.

    This is the top-level event for a dream run, providing aggregate metrics
    across all specialists. Individual specialist details are captured in
    DreamSpecialistEvent with the same run_id for correlation.
    """

    _event_type: ClassVar[str] = "dream.run"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Run identification (for correlating with specialist/iteration/tool events)
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_name: str = Field(..., description="Most recent session name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # What ran
    specialists_run: list[str] = Field(
        ..., description="Specialists executed (e.g., ['deduction', 'induction'])"
    )
    deduction_success: bool = Field(
        ..., description="Whether deduction specialist succeeded"
    )
    induction_success: bool = Field(
        ..., description="Whether induction specialist succeeded"
    )

    # Surprisal sampling (optional phase)
    surprisal_enabled: bool = Field(
        default=False, description="Whether surprisal sampling was enabled"
    )
    surprisal_conclusion_count: int = Field(
        default=0, description="High-surprisal conclusions found"
    )

    # Aggregated metrics across all specialists
    total_iterations: int = Field(
        ..., description="Total LLM iterations across all specialists"
    )
    total_input_tokens: int = Field(
        ..., description="Total input tokens across all specialists"
    )
    total_output_tokens: int = Field(
        ..., description="Total output tokens across all specialists"
    )
    total_duration_ms: float = Field(..., description="Total processing time")

    def get_resource_id(self) -> str:
        """Resource ID is the run_id for uniqueness."""
        return self.run_id


class DreamSpecialistEvent(BaseEvent):
    """Emitted when an individual dream specialist completes.

    Each specialist (deduction, induction) emits its own event with
    the same run_id as the parent DreamRunEvent for correlation.
    """

    _event_type: ClassVar[str] = "dream.specialist"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "dream"

    # Run identification (correlates with parent dream.run)
    run_id: str = Field(..., description="8-char UUID prefix for run correlation")

    # Specialist info
    specialist_type: str = Field(
        ..., description="Specialist type: 'deduction' or 'induction'"
    )

    # Workspace context
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="Workspace name")

    # Peer context
    observer: str = Field(..., description="Observer peer name")
    observed: str = Field(..., description="Observed peer name")

    # Execution metrics
    iterations: int = Field(..., description="Number of LLM iterations")
    tool_calls_count: int = Field(..., description="Total tool calls made")
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")
    duration_ms: float = Field(..., description="Processing time")
    success: bool = Field(..., description="Whether the specialist succeeded")

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and specialist type for uniqueness."""
        return f"{self.run_id}:{self.specialist_type}"


__all__ = [
    "DreamRunEvent",
    "DreamSpecialistEvent",
]
