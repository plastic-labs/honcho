from logging import getLogger
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.filter import apply_filter

logger = getLogger(__name__)


async def get_or_create_peers(
    db: AsyncSession,
    workspace_name: str,
    peers: list[schemas.PeerCreate],
    _retry: bool = False,
) -> list[models.Peer]:
    """
    Get an existing list of peers or create new peers if they don't exist.
    Updates existing peers with metadata and configuration if provided.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peers: List of peer creation schemas
        _retry: Whether to retry the operation

    Returns:
        List of peers if found or created

    Raises:
        ConflictException: If we fail to get or create the peers
    """
    peer_names = [p.name for p in peers]
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names))
    )
    result = await db.execute(stmt)
    existing_peers = list(result.scalars().all())

    # Create a mapping of peer names to peer schemas for easy lookup
    peer_schema_map = {p.name: p for p in peers}

    # Update existing peers with metadata and configuration if provided
    for existing_peer in existing_peers:
        peer_schema = peer_schema_map[existing_peer.name]

        # Update with metadata and configuration if provided
        if peer_schema.metadata is not None:
            existing_peer.h_metadata = peer_schema.metadata

        if peer_schema.configuration is not None:
            existing_peer.configuration = peer_schema.configuration

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
        # Return combined list of existing and new peers
        return existing_peers + new_peers
    except IntegrityError:
        await db.rollback()
        if _retry:
            raise ConflictException(
                f"Unable to create or get peers: {peer_names}"
            ) from None
        return await get_or_create_peers(db, workspace_name, peers, _retry=True)


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
        The peer if found or created

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    # Try to get the existing peer
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == peer.name)
    )
    result = await db.execute(stmt)
    existing_peer = result.scalar_one_or_none()

    if existing_peer is not None:
        return existing_peer

    raise ResourceNotFoundException(
        f"Peer {peer.name} not found in workspace {workspace_name}"
    )


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
    )[0]

    if peer.metadata is not None:
        honcho_peer.h_metadata = peer.metadata

    if peer.configuration is not None:
        honcho_peer.configuration = peer.configuration

    await db.commit()
    logger.info(f"Peer {peer_name} updated successfully")
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
