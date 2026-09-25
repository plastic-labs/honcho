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
    _schema_version: ClassVar[int] = 2
    _category: ClassVar[str] = "dream"

    # Run identification (for correlating with specialist/iteration/tool events)
    run_id: str = Field(..., description="Nanoid for run correlation")

    # Workspace context
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_name: str | None = Field(None, description="Session name if specified")

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

    # ---- Additive fields ----
    dream_type: str | None = Field(
        default=None,
        description="DreamType slug (currently 'omni'; future: 'deductive'/'inductive')",
    )
    enabled_types_count: int = Field(
        default=0,
        description="len(settings.DREAM.ENABLED_TYPES) at run start — how many dream types this deploy was producing",
    )
    trigger_reason: str | None = Field(
        default=None,
        description=(
            "What tripped the schedule: 'document_threshold' | 'manual' | 'surprisal'. "
            "Captured at schedule time and threaded through the queue payload."
        ),
    )
    delay_reason: str | None = Field(
        default=None,
        description=(
            "What governed when this dream actually fired: 'idle_timeout' | "
            "'immediate' | 'min_hours_gate'. Disambiguates from trigger_reason "
            "to preserve the two-gate scheduler semantics in analytics."
        ),
    )
    documents_since_last_dream_at_schedule: int | None = Field(
        default=None,
        description=(
            "Document count at the moment check_and_schedule_dream made the decision. "
            "Named _at_schedule because the live count changes between schedule and fire "
            "(idle delay) — this is the snapshot, not the current value."
        ),
    )
    document_threshold: int | None = Field(
        default=None,
        description="settings.DREAM.DOCUMENT_THRESHOLD snapshot at schedule time",
    )

    def get_resource_id(self) -> str:
        """Resource ID is the run_id for uniqueness."""
        return self.run_id


class DreamSpecialistEvent(BaseEvent):
    """Emitted when an individual dream specialist completes.

    Each specialist (deduction, induction) emits its own event with
    the same run_id as the parent DreamRunEvent for correlation.
    """

    _event_type: ClassVar[str] = "dream.specialist"
    _schema_version: ClassVar[int] = 2
    _category: ClassVar[str] = "dream"

    # Run identification (correlates with parent dream.run)
    run_id: str = Field(..., description="Nanoid for run correlation")

    # Specialist info
    specialist_type: str = Field(
        ..., description="Specialist type: 'deduction' or 'induction'"
    )

    # Workspace context
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

    # ---- Additive fields ----
    # Denormalized rollups so analytics can answer "how many observations did
    # this specialist actually produce" without re-aggregating per-tool events.
    # Sourced from ToolResult.metadata via tool_loop's all_tool_calls, NOT
    # from tool-name counting — `create_observations` calls can produce zero
    # observations when all entries fail validation.
    created_observation_count: int = Field(
        default=0,
        description="Actual observations created across all create_observations calls (from ToolResult.metadata.created_count)",
    )
    deleted_observation_count: int = Field(
        default=0,
        description="Actual observations deleted across all delete_observations calls (from ToolResult.metadata.deleted_count)",
    )
    created_counts_by_level: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Counts of created observations per level (explicit / deductive / "
            "inductive / contradiction), aggregated across all "
            "create_observations tool calls in this specialist run. Levels "
            "with zero count may be omitted; queries should treat missing "
            "keys as 0. Dict-of-counts rather than list[str] because dream "
            "specialists can produce 10-20+ observations per run — a flat "
            "list becomes noisy at that scale."
        ),
    )
    deleted_counts_by_level: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Counts of deleted observations per level, aggregated across all "
            "delete_observations tool calls in this specialist run."
        ),
    )
    peer_card_updated: bool = Field(
        default=False,
        description="True when at least one update_peer_card tool call succeeded",
    )
    search_tool_calls_count: int = Field(
        default=0,
        description="Number of search_memory / search_messages / search_messages_temporal invocations",
    )
    error_class: str | None = Field(
        default=None,
        description="Exception class name when success=False; None on success.",
    )

    def get_resource_id(self) -> str:
        """Resource ID includes run_id and specialist type for uniqueness."""
        return f"{self.run_id}:{self.specialist_type}"


__all__ = [
    "DreamRunEvent",
    "DreamSpecialistEvent",
]
