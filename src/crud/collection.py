from logging import getLogger
from typing import Any

from cashews import NOT_NONE
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src import models
from src.cache.client import (
    cache,
    get_cache_namespace,
    safe_cache_delete,
    safe_cache_set,
)
from src.config import settings
from src.exceptions import ConflictException, ResourceNotFoundException

logger = getLogger(__name__)

COLLECTION_CACHE_KEY_TEMPLATE = (
    "workspace:{workspace_name}:collection:{observer}:{observed}"
)
COLLECTION_LOCK_PREFIX = f"{get_cache_namespace()}:lock"


def collection_cache_key(workspace_name: str, observer: str, observed: str) -> str:
    """Generate cache key for collection."""
    return (
        get_cache_namespace()
        + ":"
        + COLLECTION_CACHE_KEY_TEMPLATE.format(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
    )


@cache(
    key=COLLECTION_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_TTL_SECONDS}s",
    prefix=get_cache_namespace(),
    condition=NOT_NONE,
)
@cache.locked(
    key=COLLECTION_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_LOCK_TTL_SECONDS}s",
    prefix=COLLECTION_LOCK_PREFIX,
)
async def _fetch_collection(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> dict[str, Any] | None:
    """Fetch a collection from the database and return as a plain dict for safe caching."""
    obj = await db.scalar(
        select(models.Collection)
        .where(models.Collection.workspace_name == workspace_name)
        .where(models.Collection.observer == observer)
        .where(models.Collection.observed == observed)
    )
    if obj is None:
        return None
    return {
        "id": obj.id,
        "observer": obj.observer,
        "observed": obj.observed,
        "workspace_name": obj.workspace_name,
        "h_metadata": obj.h_metadata,
        "internal_metadata": obj.internal_metadata,
        "created_at": obj.created_at,
    }


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
    data = await _fetch_collection(db, workspace_name, observer, observed)
    if data is None:
        raise ResourceNotFoundException("Collection not found")
    # Reconstruct ORM object from cached dict and merge into session
    obj = models.Collection(**data)
    make_transient_to_detached(obj)
    collection = await db.merge(obj, load=False)
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

            key = collection_cache_key(workspace_name, observer, observed)
            await safe_cache_set(
                key,
                {
                    "id": honcho_collection.id,
                    "observer": honcho_collection.observer,
                    "observed": honcho_collection.observed,
                    "workspace_name": honcho_collection.workspace_name,
                    "h_metadata": honcho_collection.h_metadata,
                    "internal_metadata": honcho_collection.internal_metadata,
                    "created_at": honcho_collection.created_at,
                },
                expire=settings.CACHE.DEFAULT_TTL_SECONDS,
            )

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


async def update_collection_internal_metadata(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    update_data: dict[str, Any],
) -> None:
    """Merge a patch into a collection's internal_metadata (JSONB ||) and invalidate cache."""
    stmt = (
        update(models.Collection)
        .where(
            models.Collection.workspace_name == workspace_name,
            models.Collection.observer == observer,
            models.Collection.observed == observed,
        )
        .values(
            internal_metadata=models.Collection.internal_metadata.op("||")(update_data)
        )
    )
    await db.execute(stmt)
    await db.commit()

    await safe_cache_delete(collection_cache_key(workspace_name, observer, observed))
