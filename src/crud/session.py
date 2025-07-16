from logging import getLogger
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import Select, cast, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import BigInteger

from src import models, schemas
from src.config import settings
from src.exceptions import ResourceNotFoundException
from src.utils.filter import apply_filter

from .peer import get_or_create_peers, get_peer

# Import workspace and peer functions that are needed
from .workspace import get_or_create_workspace

logger = getLogger(__name__)


async def get_sessions(
    workspace_name: str,
    filters: dict[str, Any] | None = None,
) -> Select[tuple[models.Session]]:
    """
    Get all sessions in a workspace.
    """
    stmt = select(models.Session).where(models.Session.workspace_name == workspace_name)

    stmt = apply_filter(stmt, models.Session, filters)

    return stmt.order_by(models.Session.created_at)


async def get_or_create_session(
    db: AsyncSession,
    session: schemas.SessionCreate,
    workspace_name: str,
) -> models.Session:
    """
    Get or create a session in a workspace with specified peers.
    If the session already exists, the peers are added to the session.

    Args:
        db: Database session
        session: Session creation schema
        workspace_name: Name of the workspace
        peer_names: List of peer names to add to the session

    Returns:
        The created session

    Raises:
        ResourceNotFoundException: If the session does not exist and create is false
    """

    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session.name)
    )

    result = await db.execute(stmt)

    honcho_session = result.scalar_one_or_none()

    # Check if session already exists
    if honcho_session is None:
        if (
            session.peer_names
            and len(session.peer_names) > settings.SESSION_PEERS_LIMIT
        ):
            raise ValueError(
                f"Cannot create session {session.name} with {len(session.peer_names)} peers. Maximum allowed is {settings.SESSION_PEERS_LIMIT} peers per session."
            )

        # Get or create workspace to ensure it exists
        await get_or_create_workspace(
            db,
            schemas.WorkspaceCreate(name=workspace_name),
        )

        # Create honcho session
        honcho_session = models.Session(
            workspace_name=workspace_name,
            name=session.name,
            h_metadata=session.metadata or {},
            configuration=session.configuration or {},
        )
        db.add(honcho_session)
        # Flush to ensure session exists in DB before adding peers
        await db.flush()
    else:
        # Update existing session with metadata and feature flags if provided
        if session.metadata is not None:
            honcho_session.h_metadata = session.metadata
        if session.configuration is not None:
            honcho_session.configuration = session.configuration

    # Add all peers to session
    if session.peer_names:
        await get_or_create_peers(
            db,
            workspace_name=workspace_name,
            peers=[
                schemas.PeerCreate(name=peer_name) for peer_name in session.peer_names
            ],
        )
        await _get_or_add_peers_to_session(
            db,
            workspace_name=workspace_name,
            session_name=session.name,
            peer_names=session.peer_names,
        )

    await db.commit()
    logger.info(
        f"Session {session.name} updated successfully in workspace {workspace_name} with {len(session.peer_names or [])} peers"
    )
    return honcho_session


async def get_session(
    db: AsyncSession,
    session_name: str,
    workspace_name: str,
) -> models.Session:
    """
    Get a session in a workspace.

    Args:
        db: Database session
        session_name: Name of the session
        workspace_name: Name of the workspace

    Returns:
        The session

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
    )

    result = await db.execute(stmt)

    honcho_session = result.scalar_one_or_none()

    if honcho_session is None:
        raise ResourceNotFoundException(
            f"Session {session_name} not found in workspace {workspace_name}"
        )

    return honcho_session


async def update_session(
    db: AsyncSession,
    session: schemas.SessionUpdate,
    workspace_name: str,
    session_name: str,
) -> models.Session:
    """
    Update a session.

    Args:
        db: Database session
        session: Session update schema
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        The updated session

    Raises:
        ResourceNotFoundException: If the session does not exist or peer is not in session
    """
    honcho_session = await get_or_create_session(
        db, schemas.SessionCreate(name=session_name), workspace_name=workspace_name
    )

    if session.metadata is not None:
        honcho_session.h_metadata = session.metadata

    if session.configuration is not None:
        honcho_session.configuration = session.configuration

    await db.commit()
    logger.info(f"Session {session_name} updated successfully")
    return honcho_session


async def delete_session(
    db: AsyncSession, workspace_name: str, session_name: str
) -> bool:
    """
    Mark a session as inactive (soft delete).

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        True if the session was deleted successfully

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
    )
    result = await db.execute(stmt)
    honcho_session = result.scalar_one_or_none()

    if honcho_session is None:
        logger.warning(
            f"Session {session_name} not found in workspace {workspace_name}"
        )
        raise ResourceNotFoundException("Session not found")

    honcho_session.is_active = False
    await db.commit()
    logger.info(f"Session {session_name} marked as inactive")
    return True


async def clone_session(
    db: AsyncSession,
    workspace_name: str,
    original_session_name: str,
    cutoff_message_id: str | None = None,
) -> models.Session:
    """
    Clone a session and its messages. If cutoff_message_id is provided,
    only clone messages up to and including that message.

    Args:
        db: SQLAlchemy session
        workspace_name: Name of the workspace the target session is in
        original_session_name: Name of the session to clone
        cutoff_message_id: Optional ID of the last message to include in the clone

    Returns:
        The newly created session
    """
    # Get the original session
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == original_session_name)
    )
    result = await db.execute(stmt)
    original_session = result.scalar_one_or_none()
    if original_session is None:
        raise ResourceNotFoundException("Original session not found")

    # If cutoff_message_id is provided, verify it belongs to the session
    cutoff_message = None
    if cutoff_message_id is not None:
        stmt = select(models.Message).where(
            models.Message.public_id == cutoff_message_id,
            models.Message.session_name == original_session_name,
        )
        cutoff_message = await db.scalar(stmt)
        if not cutoff_message:
            raise ValueError(
                "Message not found or doesn't belong to the specified session"
            )

    # Create new session
    new_session = models.Session(
        workspace_name=workspace_name,
        name=generate_nanoid(),
        h_metadata=original_session.h_metadata,
    )
    db.add(new_session)
    await db.flush()  # Flush to get the new session ID

    # Build query for messages to clone
    stmt = select(models.Message).where(
        models.Message.session_name == original_session_name
    )
    if cutoff_message_id is not None and cutoff_message is not None:
        stmt = stmt.where(models.Message.id <= cast(cutoff_message.id, BigInteger))
    stmt = stmt.order_by(models.Message.id)

    # Fetch messages to clone
    messages_to_clone_scalars = await db.scalars(stmt)
    messages_to_clone = messages_to_clone_scalars.all()

    if not messages_to_clone:
        return new_session

    # Prepare bulk insert data
    new_messages = [
        {
            "session_name": new_session.name,
            "content": message.content,
            "h_metadata": message.h_metadata,
            "workspace_name": workspace_name,
            "peer_name": message.peer_name,
        }
        for message in messages_to_clone
    ]

    insert_stmt = insert(models.Message).returning(models.Message)
    result = await db.execute(insert_stmt, new_messages)

    # Clone peers from original session to new session
    stmt = select(models.SessionPeer).where(
        models.SessionPeer.session_name == original_session_name
    )
    result = await db.execute(stmt)
    session_peers = result.scalars().all()
    for session_peer in session_peers:
        new_session_peer = models.SessionPeer(
            session_name=new_session.name,
            peer_name=session_peer.peer_name,
            workspace_name=workspace_name,
        )
        db.add(new_session_peer)

    await db.commit()
    logger.info(f"Session {original_session_name} cloned successfully")
    return new_session


async def remove_peers_from_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_names: set[str],
) -> bool:
    """
    Remove specified peers from a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_names: Set of peer names to remove from the session

    Returns:
        True if peers were removed successfully

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    # Verify session exists
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise ResourceNotFoundException(
            f"Session {session_name} not found in workspace {workspace_name}"
        )

    # Soft delete specified session peers by setting left_at timestamp
    update_stmt = (
        update(models.SessionPeer)
        .where(
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.peer_name.in_(peer_names),
            models.SessionPeer.left_at.is_(None),  # Only update active peers
        )
        .values(left_at=func.now())
    )
    result = await db.execute(update_stmt)

    await db.commit()
    return True


async def get_peers_from_session(
    workspace_name: str,
    session_name: str,
) -> Select[tuple[models.Peer]]:
    """
    Get all peers from a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        Paginated list of Peer objects in the session
    """
    # Get all active peers in the session (where left_at is NULL)
    return (
        select(models.Peer)
        .join(models.SessionPeer, models.Peer.name == models.SessionPeer.peer_name)
        .where(models.SessionPeer.session_name == session_name)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.SessionPeer.left_at.is_(None))  # Only active peers
    )


async def get_session_peer_configuration(
    workspace_name: str,
    session_name: str,
) -> Select[tuple[str, dict[str, Any], dict[str, Any]]]:
    """
    Get configuration from both SessionPeer and Peer tables for active peers in a session.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        Select statement returning peer_name, peer_configuration, and session_peer_configuration
    """
    stmt: Select[tuple[str, dict[str, Any], dict[str, Any]]] = (
        select(
            models.Peer.name.label("peer_name"),
            models.Peer.configuration.label("peer_configuration"),
            models.SessionPeer.configuration.label("session_peer_configuration"),
        )
        .join(models.SessionPeer, models.Peer.name == models.SessionPeer.peer_name)
        .where(models.SessionPeer.session_name == session_name)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.SessionPeer.workspace_name == workspace_name)
        .where(models.SessionPeer.left_at.is_(None))  # Only active peers
    )

    return stmt


async def set_peers_for_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_names: dict[str, schemas.SessionPeerConfig],
) -> list[models.SessionPeer]:
    """
    Set peers for a session, overwriting any existing peers.
    If peers don't exist, they will be created.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_names: Set of peer names to set for the session

    Returns:
        List of SessionPeer objects for all peers in the session

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    # Validate peer limit before making any changes
    if len(peer_names) > settings.SESSION_PEERS_LIMIT:
        raise ValueError(
            f"Cannot set {len(peer_names)} peers for session {session_name}. Maximum allowed is {settings.SESSION_PEERS_LIMIT} peers per session."
        )

    # Verify session exists
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise ResourceNotFoundException(
            f"Session {session_name} not found in workspace {workspace_name}"
        )

    # Soft delete specified session peers by setting left_at timestamp
    update_stmt = (
        update(models.SessionPeer)
        .where(
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.left_at.is_(None),  # Only update active peers
        )
        .values(left_at=func.now())
    )
    result = await db.execute(update_stmt)

    # Get or create peers
    await get_or_create_peers(
        db,
        workspace_name=workspace_name,
        peers=[schemas.PeerCreate(name=peer_name) for peer_name in peer_names],
    )

    # Add new peers to session
    peers = await _get_or_add_peers_to_session(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        peer_names=peer_names,
    )

    await db.commit()
    return peers


async def _get_or_add_peers_to_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_names: dict[str, schemas.SessionPeerConfig],
) -> list[models.SessionPeer]:
    """
    Add multiple peers to an existing session. If a peer already exists in the session,
    it will be skipped gracefully.

    Args:
        db: Database session
        session_name: Name of the session
        peer_names: Set of peer names to add to the session

    Returns:
        List of all SessionPeer objects (both existing and newly created)

    Raises:
        ValueError: If adding peers would exceed the maximum limit
    """
    # If no peers to add, skip the insert and just return existing active session peers
    if not peer_names:
        select_stmt = select(models.SessionPeer).where(
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.left_at.is_(None),  # Only active peers
        )
        result = await db.execute(select_stmt)
        return list(result.scalars().all())

    # Check current number of active peers and validate limit before upsert
    current_peers_stmt = select(models.SessionPeer.peer_name).where(
        models.SessionPeer.session_name == session_name,
        models.SessionPeer.workspace_name == workspace_name,
        models.SessionPeer.left_at.is_(None),  # Only active peers
    )
    result = await db.execute(current_peers_stmt)
    existing_peer_names = result.scalars().all()

    new_peers = [name for name in peer_names if name not in existing_peer_names]
    if len(new_peers) + len(existing_peer_names) > settings.SESSION_PEERS_LIMIT:
        raise ValueError(
            f"Cannot add {len(new_peers)} peer(s). Session already has {len(existing_peer_names)} peer(s) with {settings.SESSION_PEERS_LIMIT} peers per session."
        )

    # Use upsert to handle both new peers and rejoining peers
    stmt = pg_insert(models.SessionPeer).values(
        [
            {
                "session_name": session_name,
                "peer_name": peer_name,
                "workspace_name": workspace_name,
                "joined_at": func.now(),
                "left_at": None,
                "configuration": configuration.model_dump(),
            }
            for peer_name, configuration in peer_names.items()
        ]
    )

    # On conflict, update joined_at and clear left_at (rejoin scenario)
    stmt = stmt.on_conflict_do_update(
        index_elements=["session_name", "peer_name", "workspace_name"],
        set_={
            "joined_at": func.now(),
            "left_at": None,
            "configuration": stmt.excluded.configuration,
        },
    )
    await db.execute(stmt)

    # Return all active session peers after the upsert
    select_stmt = select(models.SessionPeer).where(
        models.SessionPeer.session_name == session_name,
        models.SessionPeer.workspace_name == workspace_name,
        models.SessionPeer.left_at.is_(None),  # Only active peers
    )
    result = await db.execute(select_stmt)
    return list(result.scalars().all())


async def get_peer_config(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_id: str,
) -> schemas.SessionPeerConfig:
    """
    Get the configuration for a peer in a session.


    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_id: Name of the peer


    Returns:
        Configuration for the peer

    Raises:
        ResourceNotFoundException: If the session or peer does not exist
    """
    # Get row from session_peer table
    stmt = select(models.SessionPeer).where(
        models.SessionPeer.workspace_name == workspace_name,
        models.SessionPeer.session_name == session_name,
        models.SessionPeer.peer_name == peer_id,
    )
    result = await db.execute(stmt)
    session_peer = result.scalar_one_or_none()

    if session_peer is None:
        raise ResourceNotFoundException(
            f"Session peer {peer_id} not found in session {session_name} in workspace {workspace_name}"
        )

    return schemas.SessionPeerConfig(**session_peer.configuration)


async def set_peer_config(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_name: str,
    config: schemas.SessionPeerConfig,
) -> None:
    """
    Set the configuration for a specific peer in a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_name: Name of the peer
        config: The peer configuration to set
    """
    # First, get the session and peer to ensure they exist
    await get_session(db, session_name, workspace_name)
    await get_peer(db, workspace_name, schemas.PeerCreate(name=peer_name))

    # Check if a SessionPeer entry already exists
    stmt = (
        select(models.SessionPeer)
        .where(models.SessionPeer.session_name == session_name)
        .where(models.SessionPeer.peer_name == peer_name)
        .where(models.SessionPeer.workspace_name == workspace_name)
    )
    result = await db.execute(stmt)
    session_peer = result.scalar_one_or_none()

    update_data = config.model_dump(exclude_none=True)

    if session_peer:
        # Update existing configuration
        if session_peer.configuration:
            # Create a new dictionary and update it to ensure SQLAlchemy tracks the change
            new_config = session_peer.configuration.copy()
            new_config.update(update_data)
            session_peer.configuration = new_config
        else:
            session_peer.configuration = update_data
    else:
        # Create a new SessionPeer entry
        session_peer = models.SessionPeer(
            session_name=session_name,
            peer_name=peer_name,
            workspace_name=workspace_name,
            configuration=update_data,
        )
        db.add(session_peer)

    await db.commit()
