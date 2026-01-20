"""
CloudEvents telemetry event system for Honcho.

This module provides:
- emit(): Queue an event for emission to the telemetry backend
- BaseEvent: Base class for all event types
- Event type subclasses will be added here as they are defined

Usage:
    from src.telemetry.events import emit, BaseEvent

    # Define your event class (subclasses define their own context fields)
    class MyEvent(BaseEvent):
        _event_type = "honcho.work.my_event"
        _schema_version = 1
        _category = "work"

        # Define context fields as needed (not all events have workspace context)
        workspace_id: str
        my_field: str

        def get_resource_id(self) -> str:
            # Include workspace_id in resource_id if relevant for idempotency
            return f"{self.workspace_id}:{self.my_field}"

    # Emit the event
    emit(MyEvent(
        workspace_id="ws_123",
        my_field="some_value",
    ))
"""

import logging
from typing import TYPE_CHECKING

from src.telemetry.events.base import BaseEvent, generate_event_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "emit",
    "BaseEvent",
    "generate_event_id",
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
