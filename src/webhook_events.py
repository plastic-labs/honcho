"""
Webhook event emission system for queue status changes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select

from src.dependencies import tracked_db
from src.models import Webhook
from src.schemas import WebhookEvent

logger = logging.getLogger(__name__)


class WebhookEventEmitter:
    """Emits webhook events. This places webhook events on a queue that is processed by the webhook delivery service."""

    def __init__(self):
        self._event_queue: asyncio.Queue[WebhookEvent] = asyncio.Queue()

    async def emit_event(
        self, workspace_name: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """
        Emit a webhook event.

        Args:
            workspace_name: The workspace name
            event_type: Type of event (e.g. 'queue.empty')
            data: Event data payload
        """
        try:
            async with tracked_db() as db:
                # Get active webhooks for this workspace that subscribe to this event
                stmt = select(Webhook).where(
                    Webhook.workspace_name == workspace_name,
                    Webhook.active == True,  # noqa: E712
                    Webhook.event == event_type,
                )
                result = await db.execute(stmt)
                webhooks = result.scalars().all()

                # Queue webhook events for delivery
                for webhook in webhooks:
                    event = WebhookEvent(
                        event=event_type,
                        data=data,
                        webhook_id=webhook.id,
                        timestamp=datetime.now(),
                    )
                    await self._event_queue.put(event)
                    logger.debug(
                        f"Queued webhook event {event_type} for webhook {webhook.id}"
                    )

        except Exception as e:
            logger.error(f"Failed to emit webhook event {event_type}: {e}")


# Global event emitter instance
webhook_emitter = WebhookEventEmitter()
