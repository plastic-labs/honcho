from logging import getLogger
from typing import Any, cast

from cashews import NOT_NONE
from sqlalchemy import func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src import models
from src.cache.client import (
    cache,
    cache_key_namespace,
    cache_prefix_namespace,
    safe_cache_delete,
    safe_cache_set,
)
from src.config import settings
from src.exceptions import ConflictException, ResourceNotFoundException

logger = getLogger(__name__)

COLLECTION_CACHE_KEY_TEMPLATE = (
    "v2:workspace:{workspace_name}:collection:{observer}:{observed}"
)
COLLECTION_LOCK_PREFIX = f"{cache_prefix_namespace()}:lock:v2"


def collection_cache_key(workspace_name: str, observer: str, observed: str) -> str:
    """Generate cache key for collection."""
    return (
        cache_key_namespace()
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
    prefix=cache_prefix_namespace(),
    condition=NOT_NONE,
)
@cache.locked(
    key=COLLECTION_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_LOCK_TTL_SECONDS}s",
    prefix=COLLECTION_LOCK_PREFIX,
    check_interval=settings.CACHE.LOCK_WAIT_CHECK_INTERVAL_SECONDS,
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
    with_for_update: bool = False,
) -> models.Collection:
    """
    Get a collection by observer/observed for a workspace.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observing peer (owns the collection)
        observed: Name of the observed peer
        with_for_update: If True, acquire a row-level lock (SELECT ... FOR UPDATE)
            on the collection. Bypasses the cache so the lock is actually held
            by the current transaction. Callers using this flag must wrap the
            read and subsequent write in the same transaction (the lock is
            released on commit/rollback).

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    if with_for_update:
        # Row-lock path: go direct to DB (skip cache) so the FOR UPDATE lock
        # is actually acquired on the row in the current transaction. The
        # cached dict path would return without issuing SELECT ... FOR UPDATE.
        stmt = (
            select(models.Collection)
            .where(models.Collection.workspace_name == workspace_name)
            .where(models.Collection.observer == observer)
            .where(models.Collection.observed == observed)
            .with_for_update()
        )
        collection = await db.scalar(stmt)
        if collection is None:
            raise ResourceNotFoundException("Collection not found")
        return collection

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


async def get_collection_internal_metadata_value(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    key: str,
) -> Any | None:
    """Read one top-level internal_metadata value straight from the database.

    Skips the collection cache, which can lag writes made by
    `merge_collection_internal_metadata_value`. Returns None when the
    collection or the key is missing.
    """
    return await db.scalar(
        select(models.Collection.internal_metadata[key]).where(
            models.Collection.workspace_name == workspace_name,
            models.Collection.observer == observer,
            models.Collection.observed == observed,
        )
    )


async def merge_collection_internal_metadata_value(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    key: str,
    patch: dict[str, Any],
    *,
    commit: bool = True,
    if_unchanged: tuple[str, str | None] | None = None,
) -> bool:
    """Merge `patch` into the object stored under one top-level key, server-side.

    `update_collection_internal_metadata` replaces a top-level value whole, so
    two writers of different sub-keys under the same key would overwrite each
    other. This merges one level deeper (`value || patch`) in a single UPDATE,
    so it needs no row lock: writers of different sub-keys never collide.

    With `commit=False` the update joins the caller's transaction, and the
    caller calls `invalidate_collection_cache` once that commits.

    With `if_unchanged=(subkey, value)`, the write happens only while the
    string stored under `subkey` still equals `value` (None: absent), so a
    writer that read before another reset the value can't write over it.

    Returns:
        Whether a row was written.
    """
    current = func.coalesce(
        models.Collection.internal_metadata[key], func.jsonb_build_object()
    )
    conditions = [
        models.Collection.workspace_name == workspace_name,
        models.Collection.observer == observer,
        models.Collection.observed == observed,
    ]
    if if_unchanged is not None:
        subkey, expected = if_unchanged
        stored = models.Collection.internal_metadata.op("->")(key).op("->>")(subkey)
        conditions.append(stored.is_(None) if expected is None else stored == expected)
    stmt = (
        update(models.Collection)
        .where(*conditions)
        .values(
            internal_metadata=models.Collection.internal_metadata.op("||")(
                func.jsonb_build_object(key, current.op("||")(literal(patch, JSONB)))
            )
        )
    )
    result = cast(CursorResult[Any], await db.execute(stmt))
    written = result.rowcount > 0
    if commit:
        await db.commit()
        await invalidate_collection_cache(workspace_name, observer, observed)
    return written


async def invalidate_collection_cache(
    workspace_name: str, observer: str, observed: str
) -> None:
    """Drop a collection's cached row after a write to it commits."""
    await safe_cache_delete(collection_cache_key(workspace_name, observer, observed))
