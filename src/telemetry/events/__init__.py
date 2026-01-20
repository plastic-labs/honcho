"""
CloudEvents telemetry event system for Honcho.

This module provides:
- emit(): Queue an event for emission to the telemetry backend
- BaseEvent: Base class for all event types
- Concrete event types for all Honcho operations

Event Types:
    Work events (background processing):
    - RepresentationCompletedEvent: Message batch processed, observations extracted
    - SummaryCompletedEvent: Session summary created/updated
    - DreamCompletedEvent: Memory consolidation task completed
    - ReconciliationCompletedEvent: Vector store sync/cleanup completed
    - DeletionCompletedEvent: Resource deletion completed

    Activity events (user-initiated):
    - DialecticCompletedEvent: Chat query completed

Usage:
    from src.telemetry.events import emit, RepresentationCompletedEvent

    emit(RepresentationCompletedEvent(
        workspace_id="ws_123",
        workspace_name="my_workspace",
        session_id="sess_456",
        session_name="my_session",
        observer="user_peer",
        observed="assistant_peer",
        earliest_message_id="msg_001",
        latest_message_id="msg_010",
        message_count=10,
        explicit_observation_count=5,
        deductive_observation_count=2,
        context_preparation_ms=50.0,
        llm_call_ms=1200.0,
        total_duration_ms=1300.0,
        input_tokens=5000,
        output_tokens=500,
    ))
"""

import logging

from src.telemetry.events.base import BaseEvent, generate_event_id
from src.telemetry.events.types import (
    DeletionCompletedEvent,
    DialecticCompletedEvent,
    DreamCompletedEvent,
    ReconciliationCompletedEvent,
    RepresentationCompletedEvent,
    SummaryCompletedEvent,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Core
    "emit",
    "BaseEvent",
    "generate_event_id",
    # Work events
    "RepresentationCompletedEvent",
    "SummaryCompletedEvent",
    "DreamCompletedEvent",
    "ReconciliationCompletedEvent",
    "DeletionCompletedEvent",
    # Activity events
    "DialecticCompletedEvent",
    # Lifecycle
    "initialize_telemetry_events",
    "shutdown_telemetry_events",
]


def emit(event: BaseEvent) -> None:
    """Queue an event for emission to the telemetry backend.

    This is the main entry point for emitting telemetry events.
    Events are buffered and sent asynchronously to the configured endpoint.

    If telemetry is disabled or not initialized, this is a no-op.

    Args:
        event: The event to emit (must be a BaseEvent subclass instance)

    Example:
        from src.telemetry.events import emit

        emit(MyEvent(
            my_field="some_value",
            ...
        ))
    """
    from src.telemetry.emitter import get_emitter

    emitter = get_emitter()
    if emitter is None:
        logger.debug("Telemetry emitter not initialized, dropping event")
        return

    emitter.emit(event)


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
