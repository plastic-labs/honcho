from logging import getLogger
from typing import Any

from cashews import NOT_NONE
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src import models, schemas
from src.cache.client import cache, get_cache_namespace, safe_cache_delete
from src.config import settings
from src.crud.workspace import get_or_create_workspace
from src.exceptions import ConflictException, ResourceNotFoundException
from src.models import Peer
from src.utils.filter import apply_filter
from src.utils.types import GetOrCreateResult

logger = getLogger(__name__)

PEER_CACHE_KEY_TEMPLATE = "workspace:{workspace_name}:peer:{peer_name}"
PEER_LOCK_PREFIX = f"{get_cache_namespace()}:lock"


def peer_cache_key(workspace_name: str, peer_name: str) -> str:
    """Generate cache key for peer."""
    return (
        get_cache_namespace()
        + ":"
        + PEER_CACHE_KEY_TEMPLATE.format(
            workspace_name=workspace_name,
            peer_name=peer_name,
        )
    )


async def get_or_create_peers(
    db: AsyncSession,
    workspace_name: str,
    peers: list[schemas.PeerCreate],
    *,
    _retry: bool = False,
) -> GetOrCreateResult[list[models.Peer]]:
    """
    Get an existing list of peers or create new peers if they don't exist.
    Updates existing peers with metadata and configuration if provided.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peers: List of peer creation schemas
        _retry: Whether to retry the operation

    Returns:
        GetOrCreateResult containing the list of peers and whether any were created

    Raises:
        ConflictException: If we fail to get or create the peers
    """

    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))
    peer_names = [p.name for p in peers]
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names))
    )
    result = await db.execute(stmt)
    existing_peers: list[Peer] = list(result.scalars().all())

    # Create a mapping of peer names to peer schemas for easy lookup
    peer_schema_map = {p.name: p for p in peers}

    # Track which peers actually changed
    changed_peers: list[Peer] = []

    # Update existing peers with metadata and configuration if provided
    for existing_peer in existing_peers:
        peer_schema = peer_schema_map[existing_peer.name]
        changed = False

        # Update with metadata if provided AND different
        if (
            peer_schema.metadata is not None
            and existing_peer.h_metadata != peer_schema.metadata
        ):
            existing_peer.h_metadata = peer_schema.metadata
            changed = True

        # Update with configuration if provided AND different
        if (
            peer_schema.configuration is not None
            and existing_peer.configuration != peer_schema.configuration
        ):
            existing_peer.configuration = peer_schema.configuration
            changed = True

        if changed:
            changed_peers.append(existing_peer)

    # Find which peers need to be created
    existing_names = {p.name for p in existing_peers}
    peers_to_create = [p for p in peers if p.name not in existing_names]

    # Create new peers
    new_peers = [
        models.Peer(
            workspace_name=workspace_name,
            name=p.name,
            h_metadata=p.metadata or {},
            configuration=p.configuration or {},
        )
        for p in peers_to_create
    ]
    try:
        db.add_all(new_peers)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        if _retry:
            raise ConflictException(
                f"Unable to create or get peers: {peer_names}"
            ) from None
        return await get_or_create_peers(db, workspace_name, peers, _retry=True)

    # Only invalidate cache for changed/new peers - read-through pattern
    for peer_obj in changed_peers + new_peers:
        cache_key = peer_cache_key(workspace_name, peer_obj.name)
        await safe_cache_delete(cache_key)
        logger.debug(
            "Peer %s cache invalidated in workspace %s (changed or new)",
            peer_obj.name,
            workspace_name,
        )

    # Return combined list of existing and new peers
    # created=True if any new peers were created
    return GetOrCreateResult(existing_peers + new_peers, created=len(new_peers) > 0)


@cache(
    key=PEER_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_TTL_SECONDS}s",
    prefix=get_cache_namespace(),
    condition=NOT_NONE,
)
@cache.locked(
    key=PEER_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_LOCK_TTL_SECONDS}s",
    prefix=PEER_LOCK_PREFIX,
)
async def _fetch_peer(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
) -> dict[str, Any] | None:
    """Fetch a peer from the database and return as a plain dict for safe caching."""
    obj = await db.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == peer_name)
    )
    if obj is None:
        return None
    return {
        "id": obj.id,
        "name": obj.name,
        "workspace_name": obj.workspace_name,
        "h_metadata": obj.h_metadata,
        "internal_metadata": obj.internal_metadata,
        "configuration": obj.configuration,
        "created_at": obj.created_at,
    }


async def get_peer(
    db: AsyncSession,
    workspace_name: str,
    peer: schemas.PeerCreate,
) -> models.Peer:
    """
    Get an existing peer.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer: Peer creation schema

    Returns:
        The peer if found

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    data = await _fetch_peer(db, workspace_name, peer.name)
    if data is None:
        raise ResourceNotFoundException(
            f"Peer {peer.name} not found in workspace {workspace_name}"
        )

    # Reconstruct ORM object from cached dict and merge into session
    obj = models.Peer(**data)
    make_transient_to_detached(obj)
    existing_peer = await db.merge(obj, load=False)

    return existing_peer


async def get_peers(
    workspace_name: str,
    filters: dict[str, str] | None = None,
) -> Select[tuple[models.Peer]]:
    stmt = select(models.Peer).where(models.Peer.workspace_name == workspace_name)

    stmt = apply_filter(stmt, models.Peer, filters)

    return stmt.order_by(models.Peer.created_at)


async def update_peer(
    db: AsyncSession, workspace_name: str, peer_name: str, peer: schemas.PeerUpdate
) -> models.Peer:
    """
    Update a peer.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        peer: Peer update schema

    Returns:
        The updated peer

    Raises:
        ResourceNotFoundException: If the peer does not exist
        ValidationException: If the update data is invalid
        ConflictException: If the update violates a unique constraint
    """
    honcho_peer = (
        await get_or_create_peers(
            db, workspace_name, [schemas.PeerCreate(name=peer_name)]
        )
    ).resource[0]

    needs_update = False

    if peer.metadata is not None and honcho_peer.h_metadata != peer.metadata:
        honcho_peer.h_metadata = peer.metadata
        needs_update = True

    if (
        peer.configuration is not None
        and honcho_peer.configuration != peer.configuration
    ):
        honcho_peer.configuration = peer.configuration
        needs_update = True

    # Early exit if unchanged
    if not needs_update:
        logger.debug(
            "Peer %s unchanged in workspace %s, skipping update",
            peer_name,
            workspace_name,
        )
        return honcho_peer

    await db.commit()
    await db.refresh(honcho_peer)

    cache_key = peer_cache_key(workspace_name, honcho_peer.name)
    await safe_cache_delete(cache_key)

    logger.debug("Peer %s updated successfully", peer_name)
    return honcho_peer


async def get_sessions_for_peer(
    workspace_name: str,
    peer_name: str,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Session]]:
    """
    Get all sessions for a peer through the session_peers relationship.

    Args:
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        filters: Filter sessions by metadata

    Returns:
        SQLAlchemy Select statement
    """
    stmt = (
        select(models.Session)
        .join(
            models.SessionPeer,
            (models.Session.name == models.SessionPeer.session_name)
            & (models.Session.workspace_name == models.SessionPeer.workspace_name),
        )
        .where(models.SessionPeer.peer_name == peer_name)
        .where(models.Session.workspace_name == workspace_name)
    )

    stmt = apply_filter(stmt, models.Session, filters)

    stmt: Select[tuple[models.Session]] = stmt.order_by(models.Session.created_at)

    return stmt
