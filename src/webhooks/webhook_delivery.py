"""
Webhook delivery service that processes and delivers webhook events via a proxy.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime

import httpx

from src.config import settings
from src.dependencies import tracked_db
from src.webhooks.event_emitter import webhook_emitter
from src.webhooks.events import WebhookEvent

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Service for delivering webhook events to a proxy."""

    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None
        self.shutdown_event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Start the webhook delivery service."""

        self.client = httpx.AsyncClient(
            headers={"User-Agent": "Honcho-Webhook-Service/1.0"}, timeout=30.0
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

                await self.deliver_webhook(event)

            except Exception as e:
                logger.error(f"Error in webhook delivery loop: {e}")
                await asyncio.sleep(1)

    def _generate_webhook_signature(self, payload: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload using AUTH_JWT_SECRET.

        Args:
            payload: JSON string of the webhook payload

        Returns:
            HMAC-SHA256 signature as hex string
        """
        auth_secret = os.getenv("AUTH_JWT_SECRET")
        if not auth_secret:
            raise ValueError("AUTH_JWT_SECRET not found - cannot sign webhook")

        signature = hmac.new(
            auth_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    async def deliver_webhook(self, event: WebhookEvent) -> None:
        """
        Deliver a webhook event to the configured proxy service.

        Args:
            event: The webhook event to deliver.
        """
        if not self.client:
            logger.error("Webhook proxy service is not initialized.")
            return

        try:
            webhook_urls = await self._get_workspace_endpoints(event.workspace_name)
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
                webhook_signature = self._generate_webhook_signature(event_json)
            except ValueError as e:
                logger.error(f"Failed to generate webhook signature: {e}")
                return

            proxy_payload = {
                "secret": webhook_signature,
                "event_payload": event_json,
            }

            for url in webhook_urls:
                response = await self.client.post(
                    url=url,
                    json=proxy_payload,
                    headers={"Content-Type": "application/json"},
                )

                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Successfully proxied webhook for workspace {event.workspace_name}, event {event.type} to {len(webhook_urls)} endpoints"
                    )
                else:
                    logger.error(
                        f"Failed to proxy webhook for workspace {event.workspace_name} to {len(webhook_urls)} endpoints. Proxy returned status {response.status_code}: {response.text}"
                    )

        except httpx.RequestError as e:
            logger.error(
                f"Error sending webhook for workspace {event.workspace_name} to proxy: {e}"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while delivering webhook for workspace {event.workspace_name} to proxy: {e}"
            )

    async def _get_workspace_endpoints(self, workspace_name: str) -> list[str]:
        """
        Get all webhook endpoint URLs for a workspace.

        Returns:
            List of webhook URLs for the workspace
        """
        async with tracked_db() as db:
            from src.crud.webhook import list_webhook_endpoints

            try:
                endpoints = await list_webhook_endpoints(db, workspace_name)
                urls = [endpoint.url for endpoint in endpoints]

                return urls
            except Exception as e:
                logger.error(
                    f"Error fetching webhook endpoints for {workspace_name}: {e}"
                )
                return []


# Global instance
webhook_delivery_service = WebhookDeliveryService()
