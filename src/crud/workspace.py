from logging import getLogger
from typing import Any

from cashews import NOT_NONE
from sqlalchemy import Select, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.cache.client import cache, get_cache_namespace
from src.config import settings
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.filter import apply_filter
from src.utils.types import GetOrCreateResult
from src.vector_store import get_external_vector_store

logger = getLogger(__name__)

WORKSPACE_CACHE_KEY_TEMPLATE = "workspace:{workspace_name}"
WORKSPACE_LOCK_PREFIX = f"{get_cache_namespace()}:lock"


def workspace_cache_key(workspace_name: str) -> str:
    """Generate cache key for workspace."""
    return (
        get_cache_namespace()
        + ":"
        + WORKSPACE_CACHE_KEY_TEMPLATE.format(workspace_name=workspace_name)
    )


@cache(
    key=WORKSPACE_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_TTL_SECONDS}s",
    prefix=get_cache_namespace(),
    condition=NOT_NONE,
)
@cache.locked(
    key=WORKSPACE_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_LOCK_TTL_SECONDS}s",
    prefix=WORKSPACE_LOCK_PREFIX,
)
async def _fetch_workspace(
    db: AsyncSession, workspace_name: str
) -> models.Workspace | None:
    """Fetch a workspace from the database."""
    return await db.scalar(
        select(models.Workspace).where(models.Workspace.name == workspace_name)
    )


async def get_or_create_workspace(
    db: AsyncSession,
    workspace: schemas.WorkspaceCreate,
    *,
    _retry: bool = False,
) -> GetOrCreateResult[models.Workspace]:
    """
    Get an existing workspace or create a new one if it doesn't exist.

    Args:
        db: Database session
        workspace: Workspace creation schema

    Returns:
        GetOrCreateResult containing the workspace and whether it was created

    Raises:
        ConflictException: If we fail to get or create the workspace
    """

    if not workspace.name:
        raise ValueError("Workspace name must be provided")

    # Check if workspace already exists
    existing_workspace = await _fetch_workspace(db, workspace.name)
    if existing_workspace is not None:
        # Workspace already exists
        logger.debug("Found existing workspace: %s", workspace.name)
        # Merge cached object into session (cached objects are detached)
        existing_workspace = await db.merge(existing_workspace, load=False)
        return GetOrCreateResult(existing_workspace, created=False)

    # Workspace doesn't exist, create a new one
    honcho_workspace = models.Workspace(
        name=workspace.name,
        h_metadata=workspace.metadata,
        configuration=workspace.configuration.model_dump(exclude_none=True),
    )
    try:
        db.add(honcho_workspace)
        await db.commit()
        await db.refresh(honcho_workspace)

        logger.debug("Workspace created successfully: %s", workspace.name)

        cache_key = workspace_cache_key(workspace.name)
        await cache.set(
            cache_key, honcho_workspace, expire=settings.CACHE.DEFAULT_TTL_SECONDS
        )
        return GetOrCreateResult(honcho_workspace, created=True)
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
    existing_workspace = await _fetch_workspace(db, workspace_name)

    if existing_workspace is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")

    # Merge cached object into session (cached objects are detached)
    existing_workspace = await db.merge(existing_workspace, load=False)

    return existing_workspace


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
    honcho_workspace: models.Workspace = (
        await get_or_create_workspace(
            db,
            schemas.WorkspaceCreate(
                name=workspace_name,
                metadata=workspace.metadata
                or {},  # Provide empty dict if metadata is None
            ),
        )
    ).resource

    # Track if anything changed
    needs_update = False

    if (
        workspace.metadata is not None
        and honcho_workspace.h_metadata != workspace.metadata
    ):
        honcho_workspace.h_metadata = workspace.metadata
        needs_update = True

    if workspace.configuration is not None:
        # Merge configuration instead of replacing to preserve existing keys
        base_config = (honcho_workspace.configuration or {}).copy()
        merged_config = {
            **base_config,
            **workspace.configuration.model_dump(exclude_none=True),
        }
        if honcho_workspace.configuration != merged_config:
            honcho_workspace.configuration = merged_config
            needs_update = True

    # Early exit if unchanged
    if not needs_update:
        logger.debug("Workspace %s unchanged, skipping update", workspace_name)
        return honcho_workspace

    await db.commit()
    await db.refresh(honcho_workspace)

    # Only invalidate if we actually updated
    cache_key = workspace_cache_key(workspace_name)
    await cache.delete(cache_key)

    logger.debug("Workspace with id %s updated successfully", honcho_workspace.id)
    return honcho_workspace


async def delete_workspace(db: AsyncSession, workspace_name: str) -> schemas.Workspace:
    """
    Delete a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace

    Returns:
        A snapshot of the deleted workspace as a Pydantic schema
    """
    logger.warning("Deleting workspace %s", workspace_name)
    stmt = select(models.Workspace).where(models.Workspace.name == workspace_name)
    result = await db.execute(stmt)
    honcho_workspace = result.scalar_one_or_none()

    if honcho_workspace is None:
        logger.warning("Workspace %s not found", workspace_name)
        raise ResourceNotFoundException()

    # Create a snapshot of the workspace data before deletion
    workspace_snapshot = schemas.Workspace(
        name=honcho_workspace.name,
        h_metadata=honcho_workspace.h_metadata,
        configuration=honcho_workspace.configuration,
        created_at=honcho_workspace.created_at,
    )

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
    try:
        await db.execute(
            delete(models.ActiveQueueSession).where(
                func.split_part(models.ActiveQueueSession.work_unit_key, ":", 2)
                == workspace_name
            )
        )

        # Then delete QueueItem entries
        await db.execute(
            delete(models.QueueItem).where(
                models.QueueItem.workspace_name == workspace_name
            )
        )

        # Get all collections for this workspace to delete their vector namespaces
        collections_result = await db.execute(
            select(models.Collection).where(
                models.Collection.workspace_name == workspace_name
            )
        )
        collections = collections_result.scalars().all()

        await db.execute(
            delete(models.MessageEmbedding).where(
                models.MessageEmbedding.workspace_name == workspace_name
            )
        )
        await db.execute(
            delete(models.Document).where(
                models.Document.workspace_name == workspace_name
            )
        )
        await db.execute(
            delete(models.Collection).where(
                models.Collection.workspace_name == workspace_name
            )
        )
        await db.execute(
            delete(models.Message).where(
                models.Message.workspace_name == workspace_name
            )
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
            delete(models.Session).where(
                models.Session.workspace_name == workspace_name
            )
        )
        await db.execute(
            delete(models.Peer).where(models.Peer.workspace_name == workspace_name)
        )
        await db.delete(honcho_workspace)
        await db.commit()

        # Delete vector store namespaces for this workspace
        external_vector_store = get_external_vector_store()

        # Delete message embeddings namespace for this workspace
        if external_vector_store:
            message_namespace = external_vector_store.get_vector_namespace(
                "message", workspace_name
            )
            try:
                await external_vector_store.delete_namespace(message_namespace)
                logger.debug(
                    "Deleted message embeddings namespace %s for workspace %s",
                    message_namespace,
                    workspace_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete message embeddings namespace %s: %s",
                    message_namespace,
                    e,
                )

            # Delete document embeddings namespaces for each collection
            for collection in collections:
                doc_namespace = external_vector_store.get_vector_namespace(
                    "document",
                    workspace_name,
                    collection.observer,
                    collection.observed,
                )
                try:
                    await external_vector_store.delete_namespace(doc_namespace)
                    logger.debug(
                        "Deleted document namespace %s for collection %s/%s",
                        doc_namespace,
                        collection.observer,
                        collection.observed,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to delete document namespace %s: %s",
                        doc_namespace,
                        e,
                    )

        cache_key = workspace_cache_key(workspace_name)
        workspace_pattern = f"{cache_key}*"
        await cache.delete_match(workspace_pattern)

        logger.debug("Workspace %s deleted", workspace_name)
    except Exception:
        logger.exception(
            "Failed to delete workspace %s",
            workspace_name,
        )
        await db.rollback()
        raise

    return workspace_snapshot
