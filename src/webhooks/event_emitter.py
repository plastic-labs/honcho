"""
Webhook event emission system for queue status changes.
"""

import asyncio
import logging

from .events import QueueEmptyPayload, WebhookEvent, WebhookEventType, WebhookPayload

logger = logging.getLogger(__name__)


class WebhookEventEmitter:
    """Emits webhook events. This places webhook events on a queue that is processed by the webhook delivery service."""

    def __init__(self):
        self._event_queue: asyncio.Queue[WebhookEvent] = asyncio.Queue()

    async def emit_event(
        self, workspace_name: str, event_type: WebhookEventType, data: WebhookPayload
    ) -> None:
        """
        Emit a webhook event to all endpoints for the workspace.

        Args:
            workspace_name: The workspace name
            event_type: Type of event (e.g. 'queue.empty')
            data: Event data payload
        """
        try:
            event = WebhookEvent(
                type=event_type,
                data=data,
                workspace_name=workspace_name,
            )
            await self._event_queue.put(event)

        except Exception as e:
            logger.error(f"Failed to emit webhook event {event_type}: {e}")

    async def emit_queue_empty(
        self, workspace_name: str, payload: QueueEmptyPayload
    ) -> None:
        await self.emit_event(workspace_name, WebhookEventType.QUEUE_EMPTY, payload)

    async def get_event(self, timeout: float = 1.0) -> WebhookEvent:
        """Get an event from the queue with timeout."""
        return await asyncio.wait_for(self._event_queue.get(), timeout=timeout)


# Global event emitter instance
webhook_emitter = WebhookEventEmitter()
