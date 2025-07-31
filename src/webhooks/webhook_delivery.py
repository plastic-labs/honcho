"""
Webhook delivery service that processes and delivers webhook events
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime

import httpx

from src.dependencies import tracked_db
from src.webhooks.event_emitter import webhook_emitter
from src.webhooks.events import WebhookEvent

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Service for delivering webhook events."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Start the webhook delivery service."""

        self.client = httpx.AsyncClient(
            headers={"User-Agent": "Honcho-Webhook-Service/1.0"}, timeout=30.0
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

                await self.deliver_webhook(event)

            except Exception as e:
                logger.error(f"Error in webhook delivery loop: {e}")
                await asyncio.sleep(1)

    def _generate_webhook_signature(self, payload: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload using WEBHOOK_SECRET.

        Args:
            payload: JSON string of the webhook payload

        Returns:
            HMAC-SHA256 signature as hex string
        """
        webhook_secret = os.getenv("WEBHOOK_SECRET")
        if not webhook_secret:
            raise ValueError("WEBHOOK_SECRET not found - cannot sign webhook")

        signature = hmac.new(
            webhook_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    async def deliver_webhook(self, event: WebhookEvent) -> None:
        """
        Deliver a webhook event to the configured webhook endpoints.

        Args:
            event: The webhook event to deliver.
        """
        if not self.client:
            logger.error("Webhook delivery service is not initialized.")
            return

        try:
            webhook_urls = await self._get_webhook_urls(event.workspace_name)
            if not webhook_urls:
                logger.info(
                    f"No webhook endpoints found for workspace {event.workspace_name}, skipping delivery."
                )
                return

            event_payload = {
                "type": event.type,
                "data": event.data.model_dump(),
                "timestamp": datetime.now().isoformat(),
            }

            event_json = json.dumps(
                event_payload, separators=(",", ":"), sort_keys=True
            )

            # Generate HMAC signature
            try:
                signature = self._generate_webhook_signature(event_json)
            except ValueError as e:
                logger.error(f"Failed to generate webhook signature: {e}")
                return

            for url in webhook_urls:
                response = await self.client.post(
                    url=url,
                    content=event_json,
                    headers={
                        "Content-Type": "application/json",
                        "X-Honcho-Signature": signature,
                    },
                )

                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Successfully delivered webhook for workspace {event.workspace_name}, event {event.type} to {len(webhook_urls)} endpoints"
                    )
                else:
                    logger.error(
                        f"Failed to deliver webhook for workspace {event.workspace_name} to {len(webhook_urls)} endpoints. Endpoint returned status {response.status_code}: {response.text}"
                    )

        except httpx.RequestError as e:
            logger.error(
                f"Error sending webhook for workspace {event.workspace_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while delivering webhook for workspace {event.workspace_name}: {e}"
            )

    async def _get_webhook_urls(self, workspace_name: str) -> list[str]:
        """
        Get all webhook endpoint URLs for a workspace or global endpoints.

        Returns:
            List of webhook URLs for the workspace
        """
        async with tracked_db() as db:
            from src.crud.webhook import list_webhook_endpoints

            try:
                endpoints = await list_webhook_endpoints(db, workspace_name)

                result = await db.execute(endpoints)
                endpoints = result.scalars().all()
                return [endpoint.url for endpoint in endpoints]
            except Exception as e:
                logger.error(
                    f"Error fetching webhook endpoints for {workspace_name}: {e}"
                )
                return []


# Global instance
webhook_delivery_service = WebhookDeliveryService()
