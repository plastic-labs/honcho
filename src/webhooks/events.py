import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from src.dependencies import tracked_db
from src.deriver.queue_payload import create_webhook_payload
from src.models import QueueItem

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    QUEUE_EMPTY = "queue.empty"
    TEST = "test"


class BaseWebhookEvent(BaseModel):
    """Base class for all webhook events."""

    workspace_name: str


class QueueEmptyEvent(BaseWebhookEvent):
    """Webhook event for when a queue becomes empty."""

    type: Literal[WebhookEventType.QUEUE_EMPTY] = WebhookEventType.QUEUE_EMPTY
    queue_type: str


class TestEvent(BaseWebhookEvent):
    """Webhook event for testing."""

    type: Literal[WebhookEventType.TEST] = WebhookEventType.TEST


# Union type for all webhook events
WebhookEvent = QueueEmptyEvent | TestEvent


async def publish_webhook_event(event: WebhookEvent) -> None:
    """
    Add a webhook event to our DB queue.

    Args:
        event: The webhook event to publish.
    """
    try:
        payload = create_webhook_payload(
            workspace_name=event.workspace_name,
            event_type=event.type.value,
            data=event.model_dump(mode="json"),
        )

        async with tracked_db("publish_webhook_event") as db:
            queue_item = QueueItem(
                work_unit_key=f"webhook:{event.workspace_name}",
                payload=payload,
                session_id=None,
                task_type="webhook",
            )
            db.add(queue_item)
            await db.commit()
            logger.debug(
                "Published webhook event '%s' for workspace '%s'",
                event.type,
                event.workspace_name,
            )

    except Exception as e:
        logger.exception(
            "Failed to publish webhook event %s: %s",
            event.type,
            e,
        )
