"""
Representation events for Honcho telemetry.

Representation tasks extract conclusions from messages to build peer
representations. These events track the processing of message batches
and the conclusions extracted from them.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class RepresentationCompletedEvent(BaseEvent):
    """Emitted when a representation task completes processing a message batch.

    Representation tasks extract conclusions from messages to build peer
    representations. This event captures the full context of what was processed
    and what was extracted.
    """

    _event_type: ClassVar[str] = "representation.completed"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "representation"

    # Workspace context
    workspace_name: str = Field(..., description="Workspace name")

    # Session context
    session_name: str = Field(..., description="Session name")

    # Peer context
    observed: str = Field(..., description="Observed peer name")

    # Queue processing info
    queue_items_processed: int = Field(
        ..., description="Number of QueueItem records dequeued and processed"
    )

    # Message batch info
    earliest_message_id: str = Field(..., description="First message ID in batch")
    latest_message_id: str = Field(..., description="Last message ID in batch")
    message_count: int = Field(..., description="Number of messages processed")

    # Conclusion counts
    explicit_conclusion_count: int = Field(
        ..., description="Number of explicit conclusions extracted"
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
        return f"{self.workspace_name}:{self.session_name}:{self.latest_message_id}"


__all__ = ["RepresentationCompletedEvent"]
