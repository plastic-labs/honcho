"""
Webhook delivery service that processes and delivers webhook events via a proxy.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.models import Webhook
from src.schemas import WebhookEvent

from .event_emitter import webhook_emitter

logger = logging.getLogger(__name__)


@dataclass
class WebhookCacheEntry:
    """An entry in the webhook cache."""

    webhook: Webhook
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=60)
    )

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now(timezone.utc) > self.expires_at


class WebhookDeliveryService:
    """Service for delivering webhook events to a proxy."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._webhook_cache: dict[str, WebhookCacheEntry] = {}

    async def start(self) -> None:
        """Start the webhook delivery service."""
        if not settings.WEBHOOKS.PROXY_URL:
            logger.warning(
                "Webhook proxy URL not configured. Webhook delivery is disabled."
            )
            return

        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Honcho-Webhook-Service/1.0"},
        )

        logger.info(
            f"Starting webhook delivery service, proxying to {settings.WEBHOOKS.PROXY_URL}"
        )

        # Start the delivery loop
        asyncio.create_task(self.delivery_loop())

    async def stop(self) -> None:
        """Stop the webhook delivery service."""
        logger.info("Stopping webhook delivery service")
        self.shutdown_event.set()

        if self.client:
            await self.client.aclose()
            self.client = None

    async def delivery_loop(self) -> None:
        """Main loop that processes webhook events from the queue."""
        while not self.shutdown_event.is_set():
            try:
                try:
                    event = await webhook_emitter.get_event(timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self.deliver_webhook_to_proxy(event)

            except Exception as e:
                logger.error(f"Error in webhook delivery loop: {e}")
                await asyncio.sleep(1)

    async def deliver_webhook_to_proxy(self, event: WebhookEvent) -> None:
        """
        Deliver a webhook event to the configured proxy service.

        Args:
            event: The webhook event to deliver.
        """
        if not self.client or not settings.WEBHOOKS.PROXY_URL:
            logger.error("Webhook proxy service is not initialized or configured.")
            return

        try:
            webhook = await self._get_webhook_details(event.webhook_id)
            if not webhook or not webhook.active:
                logger.warning(
                    f"Webhook {event.webhook_id} not found or inactive, skipping delivery."
                )
                return

            proxy_payload = {
                "target_url": webhook.url,
                "secret": webhook.secret,
                "event_payload": {
                    "event": event.event,
                    "data": event.data,
                    "webhook_id": event.webhook_id,
                    "timestamp": event.timestamp.isoformat(),
                },
            }

            response = await self.client.post(
                url=settings.WEBHOOKS.PROXY_URL,
                json=proxy_payload,
                headers={"Content-Type": "application/json"},
            )

            if 200 <= response.status_code < 300:
                logger.info(
                    f"Successfully proxied webhook {event.webhook_id} for event {event.event}"
                )
            else:
                logger.error(
                    f"Failed to proxy webhook {event.webhook_id}. Proxy returned status {response.status_code}: {response.text}"
                )

        except httpx.RequestError as e:
            logger.error(f"Error sending webhook {event.webhook_id} to proxy: {e}")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while delivering webhook {event.webhook_id} to proxy: {e}"
            )

    async def _get_webhook_details(self, webhook_id: str) -> Webhook | None:
        """Get webhook details from cache or database."""
        cache_entry = self._webhook_cache.get(webhook_id)
        if cache_entry and not cache_entry.is_expired:
            return cache_entry.webhook

        async with tracked_db() as db:
            webhook = await crud.get_webhook_by_id(db, webhook_id)

            if webhook:
                self._webhook_cache[webhook_id] = WebhookCacheEntry(webhook=webhook)
            return webhook


# Global webhook delivery service instance
webhook_delivery_service = WebhookDeliveryService()
