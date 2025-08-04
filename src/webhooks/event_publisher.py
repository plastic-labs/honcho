"""
Webhook event publisher that places events onto the database queue.
"""

import logging

from src.dependencies import tracked_db
from src.deriver.queue_payload import create_webhook_payload
from src.models import QueueItem
from src.webhooks.events import WebhookEvent

logger = logging.getLogger(__name__)


class WebhookQueuePublisher:
    """Places webhook events onto the database queue."""

    async def publish_event(self, event: WebhookEvent) -> None:
        """
        Create a webhook job and add it to the database queue.

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
                    payload=payload,
                    session_id=None,
                    task_type="webhook",
                )
                db.add(queue_item)
                await db.commit()
                logger.debug(
                    f"Published webhook event '{event.type}' for workspace '{event.workspace_name}'"
                )

        except Exception as e:
            logger.error(f"Failed to publish webhook event {event.type}: {e}")


# Global event publisher instance
webhook_publisher = WebhookQueuePublisher()
