from logging import getLogger

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.config import settings
from src.crud.workspace import get_workspace
from src.exceptions import ResourceNotFoundException

logger = getLogger(__name__)


async def get_or_create_webhook_endpoint(
    db: AsyncSession,
    workspace_name: str,
    webhook: schemas.WebhookEndpointCreate,
) -> schemas.WebhookEndpoint:
    """
    Get or create a webhook endpoint, optionally for a workspace.

    Args:
        db: Database session
        webhook: Webhook endpoint creation schema

    Returns:
        The webhook endpoint

    Raises:
        ResourceNotFoundException: If the workspace is specified and does not exist
    """
    # Verify workspace exists
    await get_workspace(db, workspace_name=workspace_name)

    stmt = select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.workspace_name == workspace_name,
    )
    result = await db.execute(stmt)
    endpoints = result.scalars().all()

    # No more than WORKSPACE_LIMIT webhooks per workspace
    if len(endpoints) >= settings.WEBHOOK.MAX_WORKSPACE_LIMIT:
        raise ValueError(
            f"Maximum number of webhook endpoints ({settings.WEBHOOK.MAX_WORKSPACE_LIMIT}) reached for this workspace."
        )

    # Check if webhook already exists for this workspace
    for endpoint in endpoints:
        if endpoint.url == webhook.url:
            return schemas.WebhookEndpoint.model_validate(endpoint)

    # Create new webhook endpoint
    webhook_endpoint = models.WebhookEndpoint(
        workspace_name=workspace_name,
        url=webhook.url,
    )
    db.add(webhook_endpoint)
    await db.commit()
    await db.refresh(webhook_endpoint)

    logger.debug(f"Webhook endpoint created: {webhook.url}")
    return schemas.WebhookEndpoint.model_validate(webhook_endpoint)


async def list_webhook_endpoints(
    db: AsyncSession, workspace_name: str
) -> Select[tuple[models.WebhookEndpoint]]:
    """
    List all webhook endpoints, optionally filtered by workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace (optional)

    Returns:
        List of webhook endpoints
    """
    # Verify workspace exists
    await get_workspace(db, workspace_name)

    return select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.workspace_name == workspace_name
    )


async def delete_webhook_endpoint(
    db: AsyncSession, workspace_name: str, endpoint_id: str
) -> None:
    """
    Delete a webhook endpoint.

    Args:
        db: Database session
        endpoint_id: ID of the webhook endpoint

    Raises:
        ResourceNotFoundException: If the webhook endpoint is not found
    """
    # Verify webhook endpoint exists
    stmt = select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.id == endpoint_id,
        models.WebhookEndpoint.workspace_name == workspace_name,
    )
    result = await db.execute(stmt)
    endpoint = result.scalar_one_or_none()

    if not endpoint:
        raise ResourceNotFoundException(
            f"Webhook endpoint {endpoint_id} not found for workspace {workspace_name}"
        )

    await db.delete(endpoint)
    await db.commit()

    logger.debug(f"Webhook endpoint {endpoint_id} deleted")
