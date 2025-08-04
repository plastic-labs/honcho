from enum import Enum

from pydantic import BaseModel


class WebhookEventType(str, Enum):
    QUEUE_EMPTY = "queue.empty"


class BaseWebhookEvent(BaseModel):
    """Base class for all webhook events."""

    workspace_name: str


class QueueEmptyEvent(BaseWebhookEvent):
    """Webhook event for when a queue becomes empty."""

    type: WebhookEventType = WebhookEventType.QUEUE_EMPTY
    queue_type: str


# Union type for all webhook events
WebhookEvent = QueueEmptyEvent
