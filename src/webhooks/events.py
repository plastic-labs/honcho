import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from src.dependencies import tracked_db
from src.models import QueueItem
from src.utils.queue_payload import create_webhook_payload
from src.utils.work_unit import construct_work_unit_key

logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    QUEUE_EMPTY = "queue.empty"
    TEST = "test.event"


class BaseWebhookEvent(BaseModel):
    """Base class for all webhook events."""

    workspace_id: str


class QueueEmptyEvent(BaseWebhookEvent):
    """Webhook event for when a queue becomes empty."""

    type: Literal[WebhookEventType.QUEUE_EMPTY] = WebhookEventType.QUEUE_EMPTY
    queue_type: str
    session_id: str | None = None
    observer: str | None = None
    observed: str | None = None


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
        # Note: workspace_name is no longer included in the payload
        # It's stored directly on the queue item
        payload = create_webhook_payload(
            event_type=event.type.value,
            data=event.model_dump(mode="json", exclude={"type"}),
        )

        async with tracked_db("publish_webhook_event") as db:
            queue_item = QueueItem(
                work_unit_key=construct_work_unit_key(
                    event.workspace_id,
                    {
                        "task_type": "webhook",
                    },
                ),
                payload=payload,
                session_id=None,
                task_type="webhook",
                workspace_name=event.workspace_id,
                message_id=None,  # Webhooks don't have a message_id
            )
            db.add(queue_item)
            await db.commit()
            logger.debug(
                "Published webhook event '%s' for workspace '%s'",
                event.type,
                event.workspace_id,
            )

    except Exception:
        logger.exception(
            "Failed to publish webhook event %s",
            event.type,
        )
