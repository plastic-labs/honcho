from logging import getLogger

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.exceptions import ResourceNotFoundException
from src.security import encrypt

logger = getLogger(__name__)


async def create_webhook(
    db: AsyncSession, workspace_name: str, webhook: schemas.WebhookCreate
) -> models.Webhook:
    """
    Create a new webhook.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook: Webhook creation schema

    Returns:
        The created webhook
    """
    encrypted_secret = encrypt(webhook.secret) if webhook.secret else None
    honcho_webhook = models.Webhook(
        workspace_name=workspace_name,
        url=webhook.url,
        event=webhook.event,
        secret=encrypted_secret,
        active=True,
    )
    db.add(honcho_webhook)
    await db.commit()
    logger.info(f"Webhook created successfully: {honcho_webhook.id}")
    return honcho_webhook


async def get_webhook(
    db: AsyncSession, workspace_name: str, webhook_id: str
) -> models.Webhook:
    """
    Get a webhook by ID.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook_id: ID of the webhook

    Returns:
        The webhook

    Raises:
        ResourceNotFoundException: If webhook is not found
    """
    stmt = select(models.Webhook).where(
        models.Webhook.workspace_name == workspace_name, models.Webhook.id == webhook_id
    )
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()

    if webhook is None:
        raise ResourceNotFoundException(f"Webhook with id {webhook_id} not found")

    return webhook


async def get_webhook_by_id(db: AsyncSession, webhook_id: str) -> models.Webhook | None:
    """
    Get a webhook by ID only (without workspace filtering).

    Args:
        db: Database session
        webhook_id: ID of the webhook

    Returns:
        The webhook or None if not found
    """
    stmt = select(models.Webhook).where(models.Webhook.id == webhook_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_all_webhooks(
    workspace_name: str, active_only: bool = False
) -> Select[tuple[models.Webhook]]:
    """
    Get all webhooks for a workspace.

    Args:
        workspace_name: Name of the workspace
        active_only: Only return active webhooks

    Returns:
        Select statement for webhooks
    """
    stmt = select(models.Webhook).where(models.Webhook.workspace_name == workspace_name)

    if active_only:
        stmt = stmt.where(models.Webhook.active)  # noqa: E712

    stmt = stmt.order_by(models.Webhook.created_at)
    return stmt


async def update_webhook(
    db: AsyncSession,
    workspace_name: str,
    webhook_id: str,
    webhook: schemas.WebhookUpdate,
) -> models.Webhook:
    """
    Update a webhook.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook_id: ID of the webhook
        webhook: Webhook update schema

    Returns:
        The updated webhook

    Raises:
        ResourceNotFoundException: If webhook is not found
    """
    honcho_webhook = await get_webhook(db, workspace_name, webhook_id)

    if webhook.url is not None:
        honcho_webhook.url = webhook.url
    if webhook.event is not None:
        honcho_webhook.event = webhook.event
    if webhook.secret is not None:
        honcho_webhook.secret = encrypt(webhook.secret)
    if webhook.active is not None:
        honcho_webhook.active = webhook.active

    await db.commit()
    logger.info(f"Webhook {webhook_id} updated successfully")
    return honcho_webhook


async def deactivate_webhook(
    db: AsyncSession, workspace_name: str, webhook_id: str
) -> None:
    """
    Deactivate a webhook.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        webhook_id: ID of the webhook

    Raises:
        ResourceNotFoundException: If webhook is not found
    """
    honcho_webhook = await get_webhook(db, workspace_name, webhook_id)
    honcho_webhook.active = False
    await db.commit()
    logger.info(f"Webhook {webhook_id} deactivated successfully")


async def delete_webhook(
    db: AsyncSession, workspace_name: str, webhook_id: str
) -> None:
    stmt = delete(models.Webhook).where(
        models.Webhook.workspace_name == workspace_name, models.Webhook.id == webhook_id
    )
    await db.execute(stmt)
    await db.commit()
    logger.info(f"Webhook {webhook_id} deleted successfully")
