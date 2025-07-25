"""
This module defines the event types and payload structures for webhooks.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from typing_extensions import TypedDict


class WebhookEventType(str, Enum):
    """Enum for webhook event types."""

    # Queue events
    QUEUE_EMPTY = "queue.empty"


class QueueEmptyPayload(TypedDict):
    workspace_id: str
    queue_type: str
    timestamp: datetime


WebhookPayload = QueueEmptyPayload


def validate_event_payload(
    event_type: WebhookEventType, data: dict[str, Any]
) -> WebhookPayload:
    """Validate that payload matches expected structure for event type."""
    payload_map = {
        WebhookEventType.QUEUE_EMPTY: QueueEmptyPayload,
    }
    if event_type in payload_map:
        return payload_map[event_type](**data)
    else:
        raise ValueError(f"Unknown event type: {event_type}")
