from logging import getLogger

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.dynamic_tables import (
    create_documents_table,
    drop_documents_table,
    get_documents_table_name,
)

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
    *,
    _retry: bool = False,
) -> models.Collection:
    try:
        collection = await get_collection(
            db, workspace_name, collection_name, peer_name
        )
        # Collection exists, assume its table exists (if not, migration needs to be run)
        return collection
    except ResourceNotFoundException:
        # Collection doesn't exist, create it AND its table atomically
        try:
            honcho_collection = models.Collection(
                workspace_name=workspace_name,
                peer_name=peer_name,
                name=collection_name,
            )
            db.add(honcho_collection)
            await db.flush()  # Get the ID assigned

            # Create the documents table in the same transaction
            await create_documents_table(db, honcho_collection.id)

            # Commit both collection and table creation together
            await db.commit()
            await db.refresh(honcho_collection)

            logger.info(
                f"Created collection {collection_name} and table {get_documents_table_name(honcho_collection.id)}"
            )

            return honcho_collection
        except IntegrityError:
            await db.rollback()
            if _retry:
                raise ConflictException(
                    f"Unable to create or get collection: {collection_name}"
                ) from None
            # Collection was created by another thread/process, try to get it
            return await get_or_create_collection(
                db, workspace_name, collection_name, peer_name, _retry=True
            )


async def delete_collection(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    peer_name: str,
) -> None:
    """
    Delete a collection and its associated documents table.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        collection_name: Name of the collection
        peer_name: Name of the peer

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    collection = await get_collection(db, workspace_name, collection_name, peer_name)

    # Drop the documents table first
    await drop_documents_table(db, collection.id)

    # Delete the collection
    await db.delete(collection)

    # Commit both the table drop and collection deletion
    await db.commit()
