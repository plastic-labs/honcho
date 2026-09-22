import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.crud.webhook import list_webhook_endpoints
from src.dependencies import tracked_db
from src.exceptions import WebhookTenantUnresolved
from src.utils.formatting import utc_now_iso
from src.utils.queue_payload import WebhookPayload

logger = logging.getLogger(__name__)


async def deliver_webhook(
    payload: WebhookPayload, workspace_name: str, *, tenant_id: str | None = None
) -> None:
    """Deliver a single webhook event to its configured endpoints."""
    # region ai
    # Under MULTI_TENANT, tenant_id rides in the signed body (not the URL and
    # not a header) because it is the only carrier honcho itself asserts:
    # webhook endpoint URLs are customer-registered via
    # POST /v3/workspaces/{ws}/webhooks, so a tenant could point its endpoint at
    # another tenant's URL and have the shared pool sign events into it; a
    # header sits outside the HMAC entirely, which covers exactly the body
    # bytes produced below (_generate_webhook_signature). Putting the claim in
    # the body means the existing signature already authenticates it, with no
    # signing change. The control plane's webhook ingress reads this field to
    # route a pooled event to the right tenant's webhooks.
    #
    # tenant_id is threaded explicitly by the caller rather than read from
    # tenant_context.get() at the point of use: this is the one call site that
    # builds a webhook body, so the value is auditable right here instead of
    # depending on ambient state, and it mirrors publish_webhook_event's own
    # explicit-threading of the same tenant for the same reason.
    #
    # Fails closed: under MULTI_TENANT, a work unit with no tenant_id raises
    # WebhookTenantUnresolved before any DB session is opened or any body is
    # built, rather than delivering an unattributed event the control plane
    # can only reject. QueueItem.tenant_id is stamped by every tenant-bound
    # writer, so this branch should be unreachable in practice; it exists so
    # a future writer that forgets the stamp fails loudly instead of silently.
    # endregion
    if settings.MULTI_TENANT and not tenant_id:
        raise WebhookTenantUnresolved(
            f"Webhook work unit 'webhook:{workspace_name}' has no tenant_id "
            + "but MULTI_TENANT is enabled; refusing to deliver an unattributed event."
        )
    try:
        async with tracked_db("webhook.deliver") as db:
            webhook_urls = await _get_webhook_urls(db, workspace_name)

        if not webhook_urls:
            logger.debug(
                f"No webhook endpoints for workspace {workspace_name}, skipping."
            )
            return

        event_payload: dict[str, Any] = {
            "type": payload.event_type,
            "data": payload.data,
            "timestamp": utc_now_iso(),
        }
        if settings.MULTI_TENANT:
            # tenant_id is guaranteed non-empty here: the fail-closed check
            # above already raised for the flag-on/no-tenant case.
            event_payload["tenant_id"] = tenant_id
        event_json = json.dumps(event_payload, separators=(",", ":"), sort_keys=True)

        try:
            signature = _generate_webhook_signature(event_json)
        except ValueError:
            logger.exception("Failed to generate webhook signature")
            return

        async with httpx.AsyncClient(timeout=30.0) as client:
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
                        logger.debug(
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
        logger.exception(f"Error sending webhook for {workspace_name}.")
    except Exception:
        logger.exception("Unexpected error delivering webhook.")


async def _get_webhook_urls(db: AsyncSession, workspace_name: str) -> list[str]:
    """
    Get all webhook endpoint URLs for a workspace.
    """
    try:
        endpoints = await list_webhook_endpoints(workspace_name)
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
