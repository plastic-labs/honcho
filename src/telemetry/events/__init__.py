"""
CloudEvents telemetry event system for Honcho.

This module provides:
- emit(): Queue an event for emission to the telemetry backend
- BaseEvent: Base class for all event types
- Concrete event types for all Honcho operations

Event Categories:
    representation: Message processing and conclusion extraction
    - RepresentationCompletedEvent: Message batch processed, conclusions extracted

    dream: Memory consolidation operations
    - DreamRunEvent: Full dream orchestration completed
    - DreamSpecialistEvent: Individual specialist (deduction/induction) completed

    dialectic: User-initiated queries
    - DialecticCompletedEvent: Chat query completed

    agent: Agentic loop tracking (correlates via run_id)
    - AgentIterationEvent: Per-LLM-call metrics
    - AgentToolConclusionsCreatedEvent: Conclusions created by agent
    - AgentToolConclusionsDeletedEvent: Conclusions deleted by agent
    - AgentToolPeerCardUpdatedEvent: Peer card updated by agent
    - AgentToolSummaryCreatedEvent: Summary created

    deletion: Resource removal
    - DeletionCompletedEvent: Resource deletion completed (with cascade counts)

    reconciliation: Maintenance operations
    - SyncVectorsCompletedEvent: Vector store sync completed
    - CleanupStaleItemsCompletedEvent: Stale items cleanup completed

Usage:
    from src.telemetry.events import emit, RepresentationCompletedEvent

    emit(RepresentationCompletedEvent(
        workspace_name="my_workspace",
        session_name="my_session",
        observed="assistant_peer",
        queue_items_processed=3,
        earliest_message_id="msg_001",
        latest_message_id="msg_010",
        message_count=10,
        explicit_conclusion_count=5,
        context_preparation_ms=50.0,
        llm_call_ms=1200.0,
        total_duration_ms=1300.0,
        input_tokens=5000,
        output_tokens=500,
    ))
"""

import logging

from src.telemetry.events.agent import (
    AgentIterationEvent,
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    AgentToolSummaryCreatedEvent,
)
from src.telemetry.events.base import BaseEvent, generate_event_id
from src.telemetry.events.deletion import DeletionCompletedEvent
from src.telemetry.events.dialectic import (
    DialecticCompletedEvent,
    DialecticPhaseMetrics,
)
from src.telemetry.events.dream import (
    DreamRunEvent,
    DreamSpecialistEvent,
)
from src.telemetry.events.reconciliation import (
    CleanupStaleItemsCompletedEvent,
    SyncVectorsCompletedEvent,
)
from src.telemetry.events.representation import RepresentationCompletedEvent

logger = logging.getLogger(__name__)

__all__ = [
    # Core
    "emit",
    "BaseEvent",
    "generate_event_id",
    # Representation events
    "RepresentationCompletedEvent",
    # Dream events
    "DreamRunEvent",
    "DreamSpecialistEvent",
    # Dialectic events
    "DialecticCompletedEvent",
    "DialecticPhaseMetrics",
    # Agent events
    "AgentIterationEvent",
    "AgentToolConclusionsCreatedEvent",
    "AgentToolConclusionsDeletedEvent",
    "AgentToolPeerCardUpdatedEvent",
    "AgentToolSummaryCreatedEvent",
    # Reconciliation events
    "SyncVectorsCompletedEvent",
    "CleanupStaleItemsCompletedEvent",
    # Deletion events
    "DeletionCompletedEvent",
    # Lifecycle
    "initialize_telemetry_events",
    "shutdown_telemetry_events",
]


def emit(event: BaseEvent) -> None:
    """Queue an event for emission to the telemetry backend.

    This is the main entry point for emitting telemetry events.
    Events are buffered and sent asynchronously to the configured endpoint.

    If telemetry is disabled or not initialized, this is a no-op.
    Telemetry failures are caught and logged to Sentry to avoid blocking operations.

    Args:
        event: The event to emit (must be a BaseEvent subclass instance)

    Example:
        from src.telemetry.events import emit

        emit(MyEvent(
            my_field="some_value",
            ...
        ))
    """
    try:
        from src.telemetry.emitter import get_emitter

        emitter = get_emitter()
        if emitter is None:
            logger.debug("Telemetry emitter not initialized, dropping event")
            return

        emitter.emit(event)
    except Exception as e:
        # Log to Sentry but don't block the main operation
        import sentry_sdk

        sentry_sdk.capture_exception(e)
        logger.warning(
            "Failed to emit telemetry event %s: %s",
            type(event).__name__,
            str(e),
        )


async def initialize_telemetry_events() -> None:
    """Initialize the telemetry events system based on configuration.

    This should be called once at application startup. It reads
    configuration from settings and initializes the CloudEvents emitter.

    This is typically called from initialize_telemetry() in the main
    telemetry module.
    """
    from src.config import settings
    from src.telemetry.emitter import initialize_emitter

    if not settings.TELEMETRY.ENABLED:
        logger.info("CloudEvents telemetry disabled")
        return

    await initialize_emitter(
        endpoint=settings.TELEMETRY.ENDPOINT,
        headers=settings.TELEMETRY.HEADERS,
        batch_size=settings.TELEMETRY.BATCH_SIZE,
        flush_interval_seconds=settings.TELEMETRY.FLUSH_INTERVAL_SECONDS,
        flush_threshold=settings.TELEMETRY.FLUSH_THRESHOLD,
        max_retries=settings.TELEMETRY.MAX_RETRIES,
        max_buffer_size=settings.TELEMETRY.MAX_BUFFER_SIZE,
        enabled=True,
    )

    logger.info(
        "CloudEvents telemetry initialized, endpoint: %s", settings.TELEMETRY.ENDPOINT
    )


async def shutdown_telemetry_events() -> None:
    """Shutdown the telemetry events system.

    This should be called during application shutdown to ensure
    all buffered events are flushed before exit.
    """
    from src.telemetry.emitter import shutdown_emitter

    await shutdown_emitter()
    logger.info("CloudEvents telemetry shutdown complete")
