"""
Webhook event emission system for queue status changes.
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from src.dependencies import tracked_db
from src.models import Webhook
from src.schemas import WebhookEvent

from .events import (
    QueueEmptyPayload,
    WebhookEventType,
)

logger = logging.getLogger(__name__)


class WebhookEventEmitter:
    """Emits webhook events. This places webhook events on a queue that is processed by the webhook delivery service."""

    def __init__(self):
        self._event_queue: asyncio.Queue[WebhookEvent] = asyncio.Queue()
        self._webhook_cache: dict[str, list[Webhook]] = {}
        self._cache_ttl: int = 300  # 5 minutes
        self._last_cache_update: dict[str, datetime] = {}

    async def _get_webhooks_for_event(
        self, workspace_name: str, event_type: WebhookEventType
    ) -> list[Webhook]:
        cache_key = f"{workspace_name}:{event_type.value}"

        # Check cache first
        if (
            cache_key in self._webhook_cache
            and cache_key in self._last_cache_update
            and (datetime.now() - self._last_cache_update[cache_key]).seconds
            < self._cache_ttl
        ):
            return self._webhook_cache[cache_key]

        # Cache miss - fetch from database
        async with tracked_db() as db:
            stmt = select(Webhook).where(
                Webhook.workspace_name == workspace_name,
                Webhook.active == True,  # noqa: E712
                Webhook.event == event_type,
            )
            result = await db.execute(stmt)
            webhooks = result.scalars().all()

            # Update cache
            self._webhook_cache[cache_key] = webhooks
            self._last_cache_update[cache_key] = datetime.now()

            return webhooks

    async def emit_event(
        self, workspace_name: str, event_type: WebhookEventType, data: WebhookPayload
    ) -> None:
        """
        Emit a webhook event.

        Args:
            workspace_name: The workspace name
            event_type: Type of event (e.g. 'queue.empty')
            data: Event data payload
        """
        try:
            webhooks = await self._get_webhooks_for_event(workspace_name, event_type)

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

    async def emit_queue_empty(
        self, workspace_name: str, payload: QueueEmptyPayload
    ) -> None:
        await self.emit_event(workspace_name, WebhookEventType.QUEUE_EMPTY, payload)


# Global event emitter instance
webhook_emitter = WebhookEventEmitter()
