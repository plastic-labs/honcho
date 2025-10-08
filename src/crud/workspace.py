from logging import getLogger
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def get_or_create_workspace(
    db: AsyncSession,
    workspace: schemas.WorkspaceCreate,
    *,
    _retry: bool = False,
) -> models.Workspace:
    """
    Get an existing workspace or create a new one if it doesn't exist.

    Args:
        db: Database session
        workspace: Workspace creation schema

    Returns:
        The workspace if found or created

    Raises:
        ConflictException: If we fail to get or create the workspace
    """
    # Try to get the existing workspace
    stmt = select(models.Workspace).where(models.Workspace.name == workspace.name)
    result = await db.execute(stmt)
    existing_workspace = result.scalar_one_or_none()

    if existing_workspace is not None:
        # Workspace already exists
        logger.debug(f"Found existing workspace: {workspace.name}")
        return existing_workspace

    # Workspace doesn't exist, create a new one
    honcho_workspace = models.Workspace(
        name=workspace.name,
        h_metadata=workspace.metadata,
        configuration=workspace.configuration,
    )
    try:
        db.add(honcho_workspace)
        await db.commit()
        logger.info(f"Workspace created successfully: {workspace.name}")
        return honcho_workspace
    except IntegrityError:
        await db.rollback()
        if _retry:
            raise ConflictException(
                f"Unable to create or get workspace: {workspace.name}"
            ) from None
        return await get_or_create_workspace(db, workspace, _retry=True)


async def get_all_workspaces(
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Workspace]]:
    """
    Get all workspaces.

    Args:
        db: Database session
        filters: Filter the workspaces by a dictionary of metadata
    """
    stmt = select(models.Workspace)
    stmt = apply_filter(stmt, models.Workspace, filters)
    stmt: Select[tuple[models.Workspace]] = stmt.order_by(models.Workspace.created_at)
    return stmt


async def get_workspace(
    db: AsyncSession,
    workspace_name: str,
) -> models.Workspace:
    """
    Get an existing workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace

    Returns:
        The workspace if found or created

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    # Try to get the existing peer
    stmt = select(models.Workspace).where(models.Workspace.name == workspace_name)
    result = await db.execute(stmt)
    existing_workspace = result.scalar_one_or_none()

    if existing_workspace is not None:
        return existing_workspace

    raise ResourceNotFoundException(f"Workspace {workspace_name} not found")


async def update_workspace(
    db: AsyncSession, workspace_name: str, workspace: schemas.WorkspaceUpdate
) -> models.Workspace:
    """
    Update a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        workspace: Workspace update schema

    Returns:
        The updated workspace
    """
    honcho_workspace = await get_or_create_workspace(
        db,
        schemas.WorkspaceCreate(
            name=workspace_name,
            metadata=workspace.metadata or {},  # Provide empty dict if metadata is None
        ),
    )

    if workspace.metadata is not None:
        honcho_workspace.h_metadata = workspace.metadata

    if workspace.configuration is not None:
        honcho_workspace.configuration = workspace.configuration

    await db.commit()
    logger.info(f"Workspace with id {honcho_workspace.id} updated successfully")
    return honcho_workspace


async def delete_workspace(db: AsyncSession, workspace_name: str) -> models.Workspace:
    """
    Delete a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace

    Returns:
        The deleted workspace
    """
    logger.info(f"Deleting workspace {workspace_name}")
    stmt = select(models.Workspace).where(models.Workspace.name == workspace_name)
    result = await db.execute(stmt)
    honcho_workspace = result.scalar_one_or_none()

    if honcho_workspace is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")

    # order is important here.
    # delete all active queue sessions referencing this workspace first (using work_unit_key parsing)
    # then queue items referencing this workspace

    # then embeddings
    # then documents
    # then collections
    # then messages

    # then webhook endpoints
    # then session_peers
    # then sessions
    # then peers
    # then workspace

    # Delete ActiveQueueSession entries first
    # Work unit keys have format: {task_type}:{workspace_name}:{...}
    # Extract workspace_name from position 2 (second component after splitting by ':')
    await db.execute(
        delete(models.ActiveQueueSession).where(
            func.split_part(models.ActiveQueueSession.work_unit_key, ":", 2)
            == workspace_name
        )
    )

    # Then delete QueueItem entries
    await db.execute(
        delete(models.QueueItem).where(
            func.split_part(models.QueueItem.work_unit_key, ":", 2) == workspace_name
        )
    )

    await db.execute(
        delete(models.MessageEmbedding).where(
            models.MessageEmbedding.workspace_name == workspace_name
        )
    )
    await db.execute(
        delete(models.Document).where(models.Document.workspace_name == workspace_name)
    )
    await db.execute(
        delete(models.Collection).where(
            models.Collection.workspace_name == workspace_name
        )
    )
    await db.execute(
        delete(models.Message).where(models.Message.workspace_name == workspace_name)
    )

    await db.execute(
        delete(models.WebhookEndpoint).where(
            models.WebhookEndpoint.workspace_name == workspace_name
        )
    )
    await db.execute(
        delete(models.SessionPeer).where(
            models.SessionPeer.workspace_name == workspace_name
        )
    )
    await db.execute(
        delete(models.Session).where(models.Session.workspace_name == workspace_name)
    )
    await db.execute(
        delete(models.Peer).where(models.Peer.workspace_name == workspace_name)
    )
    await db.delete(honcho_workspace)
    await db.commit()

    logger.info(f"Workspace {workspace_name} deleted")
    return honcho_workspace
