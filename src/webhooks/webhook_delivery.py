import asyncio
import hashlib
import hmac
import json
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.crud.webhook import list_webhook_endpoints
from src.utils.formatting import utc_now_iso
from src.utils.queue_payload import WebhookPayload

logger = logging.getLogger(__name__)


async def deliver_webhook(db: AsyncSession, payload: WebhookPayload) -> None:
    """
    Deliver a single webhook event to its configured endpoints.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            webhook_urls = await _get_webhook_urls(db, payload.workspace_name)
            if not webhook_urls:
                logger.info(
                    f"No webhook endpoints for workspace {payload.workspace_name}, skipping."
                )
                return

            event_payload = {
                "type": payload.event_type,
                "data": payload.data,
                "timestamp": utc_now_iso(),
            }
            event_json = json.dumps(
                event_payload, separators=(",", ":"), sort_keys=True
            )

            try:
                signature = _generate_webhook_signature(event_json)
            except ValueError:
                logger.exception("Failed to generate webhook signature")
                return

            tasks = [
                client.post(
                    url=url,
                    content=event_json,
                    headers={
                        "Content-Type": "application/json",
                        "X-Honcho-Signature": signature,
                    },
                )
                for url in webhook_urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(webhook_urls, results, strict=False):
                if isinstance(result, httpx.Response):
                    if 200 <= result.status_code < 300:
                        logger.info(
                            f"Successfully delivered webhook {payload.event_type} to {url}"
                        )
                    else:
                        logger.error(
                            f"Failed delivery for {payload.event_type} to {url}. Status: {result.status_code}"
                        )
                else:
                    logger.error(
                        f"Failed delivery for {payload.event_type} to {url}. Exception: {result}"
                    )

        except httpx.RequestError:
            logger.exception(f"Error sending webhook for {payload.workspace_name}.")
        except Exception:
            logger.exception("Unexpected error delivering webhook.")


async def _get_webhook_urls(db: AsyncSession, workspace_name: str) -> list[str]:
    """
    Get all webhook endpoint URLs for a workspace.
    """
    try:
        endpoints = await list_webhook_endpoints(db, workspace_name)
        result = await db.execute(endpoints)
        return [endpoint.url for endpoint in result.scalars().all()]
    except Exception:
        logger.exception(f"Error fetching endpoints for {workspace_name}")
        return []


def _generate_webhook_signature(payload: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload using WEBHOOK_SECRET.
    """
    webhook_secret = settings.WEBHOOK.SECRET
    if not webhook_secret:
        raise ValueError("WEBHOOK_SECRET not found - cannot sign webhook")

    return hmac.new(
        webhook_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
