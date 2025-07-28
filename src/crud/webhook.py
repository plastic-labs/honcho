from datetime import datetime
from logging import getLogger

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from src import models, schemas
from src.crud.workspace import get_workspace
from src.exceptions import ResourceNotFoundException
from src.models import WebhookStatus

logger = getLogger(__name__)


async def create_webhook_endpoint(
    db: AsyncSession,
    workspace_name: str,
    webhook: schemas.WebhookEndpointCreate,
) -> schemas.WebhookEndpoint:
    """
    Set the webhook endpoint URL for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook_endpoint: Webhook endpoint creation schema

    Returns:
        The existing webhook configuration

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """

    url, events = webhook.url, webhook.events

    await update_webhook_endpoint(db, workspace_name, str(url))

    # Create new webhook events
    new_events = [
        models.Webhook(
            workspace_name=workspace_name,
            event=event,
            status=WebhookStatus.ENABLED,
            created_at=func.now(),
        )
        for event in events
    ]
    db.add_all(new_events)

    await db.commit()
    logger.info(f"Webhook endpoint set for workspace {workspace_name}: {url}")
    return schemas.WebhookEndpoint(
        url=str(url),
        events=[schemas.Webhook.model_validate(obj=webhook) for webhook in new_events],
    )


async def add_webhook_event(
    db: AsyncSession, workspace_name: str, webhook: schemas.WebhookCreate
) -> models.Webhook:
    """
    Create or enable a webhook event subscription.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook: Webhook creation schema

    Returns:
        The created or updated webhook
    """

    workspace = await get_workspace(db, workspace_name)
    if workspace.webhook_url is None:
        raise ResourceNotFoundException(
            f"Webhook endpoint not set for workspace {workspace_name}"
        )

    # Check if webhook already exists
    stmt = select(models.Webhook).where(
        models.Webhook.workspace_name == workspace_name,
        models.Webhook.event == webhook.event,
    )
    result = await db.execute(stmt)
    existing_webhook = result.scalar_one_or_none()

    if existing_webhook:
        if existing_webhook.status == WebhookStatus.DISABLED:
            # Enable existing webhook
            existing_webhook.status = WebhookStatus.ENABLED
            existing_webhook.disabled_at = None
            await db.commit()
            logger.debug(f"Webhook re-enabled: {workspace_name}/{webhook.event}")
        return existing_webhook

    # Create new webhook
    honcho_webhook = models.Webhook(
        workspace_name=workspace_name,
        event=webhook.event,
        status=WebhookStatus.ENABLED,
    )
    db.add(honcho_webhook)
    await db.commit()
    logger.debug(f"Webhook created: {workspace_name}/{webhook.event}")
    return honcho_webhook


async def get_webhook_configuration(
    db: AsyncSession, workspace_name: str
) -> schemas.WebhookEndpoint:
    """
    Get webhook endpoint configuration for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace

    Returns:
        Webhook endpoint with events
    """
    from src.crud.workspace import get_workspace

    workspace = await get_workspace(db, workspace_name)

    # Get all webhook subscriptions
    stmt = (
        select(models.Webhook)
        .where(models.Webhook.workspace_name == workspace_name)
        .where(models.Webhook.status == WebhookStatus.ENABLED)
        .order_by(models.Webhook.created_at)
    )
    result = await db.execute(stmt)
    webhooks = result.scalars().all()

    return schemas.WebhookEndpoint(url=workspace.webhook_url, events=list(webhooks))


async def update_webhook_endpoint(
    db: AsyncSession, workspace_name: str, url: str
) -> schemas.WebhookEndpoint:
    """
    Update webhook endpoint URL for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        url: New webhook URL

    Returns:
        The updated webhook endpoint configuration

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    workspace = await get_workspace(db, workspace_name)
    workspace.webhook_url = url

    await db.commit()
    logger.info(f"Webhook endpoint updated for workspace {workspace_name}: {url}")

    # Get all webhook subscriptions
    stmt = (
        select(models.Webhook)
        .where(models.Webhook.workspace_name == workspace_name)
        .order_by(models.Webhook.created_at)
    )
    result = await db.execute(stmt)
    webhooks = result.scalars().all()

    return schemas.WebhookEndpoint(
        url=url,
        events=[schemas.Webhook.model_validate(obj=webhook) for webhook in webhooks],
    )


async def update_webhook_status(
    db: AsyncSession, workspace_name: str, webhook_update: schemas.WebhookUpdate
) -> models.Webhook:
    """
    Update webhook event status (enable/disable).

    Args:
        db: Database session
        workspace_name: Name of the workspace
        event: The webhook event type
        enabled: Whether to enable or disable the webhook

    Returns:
        The updated webhook

    Raises:
        ResourceNotFoundException: If webhook is not found
    """
    workspace = await get_workspace(db, workspace_name)
    if workspace.webhook_url is None:
        raise ResourceNotFoundException(
            f"Webhook endpoint not set for workspace {workspace_name}"
        )

    event, status = webhook_update.event, webhook_update.status
    # Find existing webhook
    stmt = select(models.Webhook).where(
        models.Webhook.workspace_name == workspace_name,
        models.Webhook.event == event,
    )
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()

    if not webhook:
        raise ResourceNotFoundException(
            f"Webhook event '{event}' not found for workspace {workspace_name}"
        )

    webhook.status = (
        WebhookStatus.ENABLED
        if status == WebhookStatus.ENABLED
        else WebhookStatus.DISABLED
    )
    webhook.disabled_at = None if status == WebhookStatus.ENABLED else datetime.now()

    await db.commit()
    logger.debug(
        f"Webhook {workspace_name}/{event} updated to {'enabled' if status == WebhookStatus.ENABLED else 'disabled'}"
    )
    return webhook


async def delete_webhook_endpoint(db: AsyncSession, workspace_name: str) -> None:
    """
    Remove webhook endpoint URL for a workspace and disable all events.

    Args:
        db: Database session
        workspace_name: Name of the workspace
    """
    from src.crud.workspace import get_workspace

    workspace = await get_workspace(db, workspace_name)
    workspace.webhook_url = None

    # Disable all webhook events
    stmt = select(models.Webhook).where(
        models.Webhook.workspace_name == workspace_name,
        models.Webhook.status == WebhookStatus.ENABLED,
    )
    result = await db.execute(stmt)
    webhooks = result.scalars().all()

    for webhook in webhooks:
        webhook.status = WebhookStatus.DISABLED
        webhook.disabled_at = func.now()

    await db.commit()
    logger.info(f"Webhook endpoint removed for workspace {workspace_name}")
