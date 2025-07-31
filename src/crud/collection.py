from logging import getLogger

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.exceptions import ConflictException, ResourceNotFoundException

logger = getLogger(__name__)


async def get_collection(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    peer_name: str | None = None,
) -> models.Collection:
    """
    Get a collection by name for a specific peer and workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.workspace_name == workspace_name)
        .where(models.Collection.name == collection_name)
    )
    if peer_name:
        stmt = stmt.where(models.Collection.peer_name == peer_name)
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        raise ResourceNotFoundException(
            "Collection not found or does not belong to peer"
        )
    return collection


async def get_or_create_collection(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    peer_name: str | None = None,
) -> models.Collection:
    try:
        return await get_collection(db, workspace_name, collection_name, peer_name)
    except ResourceNotFoundException:
        try:
            honcho_collection = models.Collection(
                workspace_name=workspace_name,
                peer_name=peer_name,
                name=collection_name,
            )
            db.add(honcho_collection)
            await db.commit()
            return honcho_collection
        except IntegrityError:
            await db.rollback()
            logger.debug(
                f"Race condition detected for collection: {collection_name}, retrying get"
            )
            try:
                return await get_collection(
                    db, workspace_name, collection_name, peer_name
                )
            except ResourceNotFoundException:
                raise ConflictException(
                    f"Unable to create or get collection: {collection_name}"
                ) from None
