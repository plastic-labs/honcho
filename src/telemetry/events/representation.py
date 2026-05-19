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
    _schema_version: ClassVar[int] = 2
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
    input_tokens: int = Field(
        ...,
        description=(
            "Queued-message tokens (the ones we're actually reasoning ABOUT). "
            "This field is the downstream metering key for "
            "representation.completed — DO NOT rename or repurpose without "
            "coordinating with downstream consumers."
        ),
    )
    total_input_tokens: int = Field(
        ...,
        description="Total tokens sent to the LLM (queued + extra context + scaffold)",
    )
    output_tokens: int = Field(..., description="Output tokens generated")

    # ---- Additive fields ----
    # Token breakdown beyond `input_tokens` (queued-message tokens already
    # captured above). These break out what made up the LLM prompt so analytics
    # can answer "how much did extra context cost us per call".
    queued_message_count: int = Field(
        default=0,
        description="Number of messages in this batch that were the actual queue items being reasoned about",
    )
    prompt_message_count: int = Field(
        default=0,
        description="Total messages in the prompt — queued + extra interleaving context",
    )
    prompt_message_tokens: int = Field(
        default=0,
        description="Sum of token_count across all messages in the prompt",
    )
    extra_context_message_count: int = Field(
        default=0,
        description="prompt_message_count - queued_message_count: the extra-context messages we pulled in",
    )
    extra_context_tokens: int = Field(
        default=0,
        description="prompt_message_tokens - input_tokens: token cost of the extra context",
    )
    prompt_scaffold_tokens: int = Field(
        default=0,
        description="Estimated tokens for the system/scaffold portion of the prompt",
    )

    # Cap configuration + hit flags ()
    batch_max_tokens: int = Field(
        default=0,
        description="settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS at fetch time",
    )
    max_input_tokens: int = Field(
        default=0, description="settings.DERIVER.MAX_INPUT_TOKENS at call time"
    )
    was_flush_enabled: bool = Field(
        default=False,
        description="settings.DERIVER.FLUSH_ENABLED snapshot at batch time",
    )
    hit_batch_token_cap: bool = Field(
        default=False,
        description="True when the queue batcher clamped the batch to fit batch_max_tokens",
    )
    hit_input_token_cap: bool = Field(
        default=False,
        description=(
            "True when the LLM call truncated input messages to fit max_input_tokens."
        ),
    )

    # Observer fanout
    observer_count: int = Field(
        default=0,
        description="Number of observers this representation was saved against",
    )

    def get_resource_id(self) -> str:
        """Resource ID includes workspace, session, and latest message for uniqueness."""
        return f"{self.workspace_name}:{self.session_name}:{self.latest_message_id}"


__all__ = ["RepresentationCompletedEvent"]
