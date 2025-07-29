from logging import getLogger

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.crud.workspace import get_workspace
from src.exceptions import ResourceNotFoundException

logger = getLogger(__name__)


async def create_webhook_endpoint(
    db: AsyncSession,
    workspace_name: str,
    webhook: schemas.WebhookEndpointCreate,
) -> schemas.WebhookEndpoint:
    """
    Create a new webhook endpoint for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook: Webhook endpoint creation schema

    Returns:
        The created webhook endpoint

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Verify workspace exists
    await get_workspace(db, workspace_name)

    # Create new webhook endpoint
    webhook_endpoint = models.WebhookEndpoint(
        workspace_name=workspace_name,
        url=webhook.url,
    )
    db.add(webhook_endpoint)
    await db.commit()
    await db.refresh(webhook_endpoint)

    logger.info(
        f"Webhook endpoint created for workspace {workspace_name}: {webhook.url}"
    )
    return schemas.WebhookEndpoint.model_validate(webhook_endpoint)


async def list_webhook_endpoints(
    db: AsyncSession, workspace_name: str
) -> list[schemas.WebhookEndpoint]:
    """
    List all webhook endpoints for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace

    Returns:
        List of webhook endpoints
    """
    # Verify workspace exists
    await get_workspace(db, workspace_name)

    stmt = select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.workspace_name == workspace_name
    )
    result = await db.execute(stmt)
    endpoints = result.scalars().all()

    return [schemas.WebhookEndpoint.model_validate(endpoint) for endpoint in endpoints]


async def delete_webhook_endpoint(
    db: AsyncSession, workspace_name: str, endpoint_id: str
) -> None:
    """
    Delete a webhook endpoint.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        endpoint_id: ID of the webhook endpoint

    Raises:
        ResourceNotFoundException: If the webhook endpoint is not found
    """

    # Verify workspace exists
    await get_workspace(db, workspace_name)

    # Verify webhook endpoint exists
    stmt = select(models.WebhookEndpoint).where(
        models.WebhookEndpoint.workspace_name == workspace_name,
        models.WebhookEndpoint.id == endpoint_id,
    )
    result = await db.execute(stmt)
    endpoint = result.scalar_one_or_none()

    if not endpoint:
        raise ResourceNotFoundException(
            f"Webhook endpoint {endpoint_id} not found for workspace {workspace_name}"
        )

    await db.delete(endpoint)
    await db.commit()

    logger.info(
        f"Webhook endpoint {endpoint_id} deleted for workspace {workspace_name}"
    )
