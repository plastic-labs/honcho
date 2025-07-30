from logging import getLogger

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_workspace
from src.exceptions import ResourceNotFoundException

logger = getLogger(__name__)


async def get_or_create_webhook_endpoint(
    db: AsyncSession,
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
    # Verify workspace exists if specified
    if webhook.workspace_name:
        await get_workspace(db, webhook.workspace_name)

    # Check if endpoint already exists
    stmt = select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.url == webhook.url,
    )
    if webhook.workspace_name:
        stmt = stmt.where(
            models.WebhookEndpoint.workspace_name == webhook.workspace_name
        )

    result = await db.execute(stmt)
    existing_endpoint = result.scalar_one_or_none()
    if existing_endpoint:
        return schemas.WebhookEndpoint.model_validate(existing_endpoint)

    # Create new webhook endpoint
    webhook_endpoint = models.WebhookEndpoint(
        workspace_name=webhook.workspace_name,
        url=webhook.url,
    )
    db.add(webhook_endpoint)
    await db.commit()
    await db.refresh(webhook_endpoint)

    logger.info(f"Webhook endpoint created: {webhook.url}")
    return schemas.WebhookEndpoint.model_validate(webhook_endpoint)


async def list_webhook_endpoints(
    db: AsyncSession, workspace_name: str | None = None
) -> list[schemas.WebhookEndpoint]:
    """
    List all webhook endpoints, optionally filtered by workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace (optional)

    Returns:
        List of webhook endpoints
    """
    # Verify workspace exists if specified
    if workspace_name:
        await get_workspace(db, workspace_name)

    # Build query
    stmt = select(models.WebhookEndpoint)
    if workspace_name:
        stmt = stmt.where(models.WebhookEndpoint.workspace_name == workspace_name)

    result = await db.execute(stmt)
    endpoints = result.scalars().all()

    return [schemas.WebhookEndpoint.model_validate(endpoint) for endpoint in endpoints]


async def delete_webhook_endpoint(db: AsyncSession, endpoint_id: str) -> None:
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
    )
    result = await db.execute(stmt)
    endpoint = result.scalar_one_or_none()

    if not endpoint:
        raise ResourceNotFoundException(f"Webhook endpoint {endpoint_id} not found")

    await db.delete(endpoint)
    await db.commit()

    logger.info(f"Webhook endpoint {endpoint_id} deleted")
