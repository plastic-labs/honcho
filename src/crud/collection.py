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
    *,
    observer: str,
    observed: str,
) -> models.Collection:
    """
    Get a collection by observer/observed for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observing peer (owns the collection)
        observed: Name of the observed peer

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.workspace_name == workspace_name)
        .where(models.Collection.observer == observer)
        .where(models.Collection.observed == observed)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        raise ResourceNotFoundException("Collection not found")
    return collection


async def get_or_create_collection(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    _retry: bool = False,
) -> models.Collection:
    try:
        return await get_collection(
            db, workspace_name, observer=observer, observed=observed
        )
    except ResourceNotFoundException:
        try:
            honcho_collection = models.Collection(
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )
            db.add(honcho_collection)
            await db.commit()
            return honcho_collection
        except IntegrityError:
            await db.rollback()
            if _retry:
                raise ConflictException(
                    f"Unable to create or get collection: {observer}/{observed}"
                ) from None
            return await get_or_create_collection(
                db, workspace_name, observer=observer, observed=observed, _retry=True
            )
