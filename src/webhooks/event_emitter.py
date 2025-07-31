"""
Webhook event emission system that places events onto the database queue.
"""

import logging

from sqlalchemy import insert

from src.dependencies import tracked_db
from src.deriver.queue_payload import create_webhook_payload
from src.models import QueueItem
from src.webhooks.events import WebhookEventType, WebhookPayload

logger = logging.getLogger(__name__)


class WebhookEventEmitter:
    """Places webhook events onto the database queue."""

    async def emit_event(
        self, workspace_name: str, event_type: WebhookEventType, data: WebhookPayload
    ) -> None:
        """
        Create a webhook job and add it to the database queue.

        Args:
            workspace_name: The workspace name.
            event_type: Type of event (e.g. 'queue.empty').
            data: Event data payload.
        """
        try:
            payload = create_webhook_payload(
                workspace_name=workspace_name,
                event_type=event_type.value,
                data=data.model_dump(mode="json"),
            )

            async with tracked_db("emit_webhook_event") as db:
                await db.execute(
                    insert(QueueItem).values(
                        payload=payload,
                        session_id=None,
                        task_type="webhook",
                    )
                )
                await db.commit()
                logger.debug(
                    f"Emitted webhook event '{event_type}' for workspace '{workspace_name}'"
                )

        except Exception as e:
            logger.error(f"Failed to emit webhook event {event_type}: {e}")


# Global event emitter instance
webhook_emitter = WebhookEventEmitter()
