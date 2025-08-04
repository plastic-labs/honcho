"""
Webhook delivery service that processes and delivers webhook events from the queue.
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.deriver.queue_payload import WebhookQueuePayload

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """Service for delivering webhook events from a queue payload."""

    def _generate_webhook_signature(self, payload: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload using WEBHOOK_SECRET.
        """
        webhook_secret = settings.WEBHOOKS.WEBHOOK_SECRET
        if not webhook_secret:
            raise ValueError("WEBHOOK_SECRET not found - cannot sign webhook")

        return hmac.new(
            webhook_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    async def deliver_webhook(
        self, db: AsyncSession, client: httpx.AsyncClient, payload: WebhookQueuePayload
    ) -> None:
        """
        Deliver a single webhook event to its configured endpoints.
        """
        try:
            webhook_urls = await self._get_webhook_urls(db, payload.workspace_name)
            if not webhook_urls:
                logger.info(
                    f"No webhook endpoints for workspace {payload.workspace_name}, skipping."
                )
                return

            event_payload = {
                "type": payload.event_type,
                "data": payload.data,
                "timestamp": datetime.now().isoformat(),
            }
            event_json = json.dumps(
                event_payload, separators=(",", ":"), sort_keys=True
            )

            try:
                signature = self._generate_webhook_signature(event_json)
            except ValueError:
                logger.exception("Failed to generate webhook signature")
                return

            for url in webhook_urls:
                response = await client.post(
                    url=url,
                    content=event_json,
                    headers={
                        "Content-Type": "application/json",
                        "X-Honcho-Signature": signature,
                    },
                )
                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Successfully delivered webhook {payload.event_type} to {url}"
                    )
                else:
                    logger.error(
                        f"Failed delivery for {payload.event_type} to {url}. Status: {response.status_code}"
                    )

        except httpx.RequestError as e:
            logger.error(f"Error sending webhook for {payload.workspace_name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error delivering webhook: {e}")

    async def _get_webhook_urls(
        self, db: AsyncSession, workspace_name: str
    ) -> list[str]:
        """
        Get all webhook endpoint URLs for a workspace.
        """
        from src.crud.webhook import list_webhook_endpoints

        try:
            endpoints = await list_webhook_endpoints(db, workspace_name)
            result = await db.execute(endpoints)
            return [endpoint.url for endpoint in result.scalars().all()]
        except Exception:
            logger.exception(f"Error fetching endpoints for {workspace_name}")
            return []


# Global instance
webhook_delivery_service = WebhookDeliveryService()
