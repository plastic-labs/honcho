from collections.abc import Sequence
from logging import getLogger
from typing import Optional

from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from openai import AsyncOpenAI
from sqlalchemy import Select, cast, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import BigInteger

from . import models, schemas
from .exceptions import (
    ConflictException,
    ResourceNotFoundException,
)

load_dotenv(override=True)

openai_client = AsyncOpenAI()

logger = getLogger(__name__)

DEF_PROTECTED_COLLECTION_NAME = "honcho"

########################################################
# workspace methods
########################################################


async def get_or_create_workspace(
    db: AsyncSession, workspace: schemas.WorkspaceCreate
) -> models.Workspace:
    """
    Get an existing workspace or create a new one if it doesn't exist.

    Args:
        db: Database session
        workspace: Workspace creation schema

    Returns:
        The workspace if found or created

    Raises:
        ConflictException: If there's an integrity error when creating the workspace
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
    try:
        honcho_workspace = models.Workspace(
            name=workspace.name,
            h_metadata=workspace.metadata,
            feature_flags=workspace.feature_flags,
        )
        db.add(honcho_workspace)
        await db.commit()
        logger.info(f"Workspace created successfully: {workspace.name}")
        return honcho_workspace
    except IntegrityError as e:
        await db.rollback()
        logger.error(
            f"IntegrityError creating workspace with name '{workspace.name}': {str(e)}"
        )
        raise ConflictException(
            f"Workspace with name '{workspace.name}' already exists"
        ) from e


async def get_all_workspaces(
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all workspaces.

    Args:
        db: Database session
        reverse: Whether to reverse the order of the workspaces
        filter: Filter the workspaces by a dictionary of metadata
    """
    stmt = select(models.Workspace)
    if reverse:
        stmt = stmt.order_by(models.Workspace.id.desc())
    else:
        stmt = stmt.order_by(models.Workspace.id)
    if filter is not None:
        stmt = stmt.where(models.Workspace.h_metadata.contains(filter))
    return stmt


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

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    try:
        honcho_workspace = await get_or_create_workspace(
            db,
            schemas.WorkspaceCreate(
                name=workspace_name,
                metadata=workspace.metadata
                or {},  # Provide empty dict if metadata is None
            ),
        )

        if workspace.metadata is not None:
            honcho_workspace.h_metadata = workspace.metadata

        if workspace.feature_flags is not None:
            honcho_workspace.feature_flags = workspace.feature_flags

        await db.commit()
        logger.info(f"Workspace with id {honcho_workspace.id} updated successfully")
        return honcho_workspace
    except IntegrityError as e:
        await db.rollback()
        logger.error(
            f"IntegrityError updating workspace {honcho_workspace.id}: {str(e)}"
        )
        raise ConflictException(
            "Workspace update failed - unique constraint violation"
        ) from e


########################################################
# peer methods
########################################################


async def get_or_create_peers(
    db: AsyncSession,
    workspace_name: str,
    peers: list[schemas.PeerCreate],
) -> list[models.Peer]:
    """
    Get an existing list of peers or create new peers if they don't exist.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peers: List of peer creation schemas

    Returns:
        List of peers if found or created
    """
    peer_names = [p.name for p in peers]
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names))
    )
    result = await db.execute(stmt)
    existing_peers = list(result.scalars().all())

    # If all peers exist, return them
    if len(existing_peers) == len(peers):
        return existing_peers

    # Find which peers need to be created
    existing_names = {p.name for p in existing_peers}
    peers_to_create = [p for p in peers if p.name not in existing_names]

    # Create new peers
    new_peers = [
        models.Peer(
            workspace_name=workspace_name,
            name=p.name,
            h_metadata=p.metadata,
            feature_flags=p.feature_flags,
        )
        for p in peers_to_create
    ]
    db.add_all(new_peers)
    await db.flush()

    # Return combined list of existing and new peers
    return existing_peers + new_peers


async def get_or_create_peer(
    db: AsyncSession,
    workspace_name: str,
    peer: schemas.PeerCreate,
    *,
    create: bool = True,
) -> models.Peer:
    """
    Get an existing peer or create a new one if it doesn't exist.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer: Peer creation schema

    Returns:
        The peer if found or created

    Raises:
        ConflictException: If there's an integrity error when creating the peer
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
        # Peer already exists
        logger.debug(f"Found existing peer: {peer.name} for workspace {workspace_name}")
        return existing_peer

    if not create:
        raise ResourceNotFoundException(
            f"Peer {peer.name} not found in workspace {workspace_name}"
        )

    # Peer doesn't exist, create a new one
    try:
        honcho_peer = models.Peer(
            workspace_name=workspace_name,
            name=peer.name,
            h_metadata=peer.metadata,
            feature_flags=peer.feature_flags,
        )
        db.add(honcho_peer)
        await db.commit()
        logger.debug(
            f"Peer created successfully: {peer.name} for workspace {workspace_name}"
        )
        return honcho_peer
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create peer - integrity error: {str(e)}")
        raise ConflictException("Peer with this name already exists") from e


async def get_peers(
    workspace_name: str,
    reverse: bool = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = select(models.Peer).where(models.Peer.workspace_name == workspace_name)

    if filter is not None:
        stmt = stmt.where(models.Peer.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Peer.id.desc())
    else:
        stmt = stmt.order_by(models.Peer.id)

    return stmt


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
    try:
        # get_peer will raise ResourceNotFoundException if not found
        honcho_peer = await get_or_create_peer(
            db, workspace_name, schemas.PeerCreate(name=peer_name)
        )

        if peer.metadata is not None:
            honcho_peer.h_metadata = peer.metadata

        if peer.feature_flags is not None:
            honcho_peer.feature_flags = peer.feature_flags

        await db.commit()
        logger.info(f"Peer {peer_name} updated successfully")
        return honcho_peer
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Peer update failed due to integrity error: {str(e)}")
        raise ConflictException(
            "Peer update failed - unique constraint violation"
        ) from e


async def get_sessions_for_peer(
    workspace_name: str,
    peer_name: str,
    reverse: bool = False,
    is_active: Optional[bool] = None,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all sessions for a peer through the session_peers relationship.

    Args:
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        reverse: Whether to reverse the order of the sessions
        is_active: Filter by active status (True/False/None for all)
        filter: Filter sessions by metadata

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

    if is_active is not None:
        stmt = stmt.where(models.Session.is_active == is_active)

    if filter is not None:
        stmt = stmt.where(models.Session.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Session.id.desc())
    else:
        stmt = stmt.order_by(models.Session.id)

    return stmt


########################################################
# session methods
########################################################


async def get_sessions(
    workspace_name: str,
    reverse: Optional[bool] = False,
    is_active: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all sessions in a workspace.
    """
    stmt = select(models.Session).where(models.Session.workspace_name == workspace_name)

    if is_active:
        stmt = stmt.where(models.Session.is_active.is_(True))

    if filter is not None:
        stmt = stmt.where(models.Session.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Session.id.desc())
    else:
        stmt = stmt.order_by(models.Session.id)

    return stmt


async def get_or_create_session(
    db: AsyncSession,
    session: schemas.SessionCreate,
    workspace_name: str,
    *,
    create: bool = True,
) -> models.Session:
    """
    Get or create a session in a workspace with specified peers.
    If the session already exists, the peers are added to the session.

    Args:
        db: Database session
        session: Session creation schema
        workspace_name: Name of the workspace
        peer_names: List of peer names to add to the session
        create: Whether to create the session if it doesn't exist. If false,
                this will raise a ResourceNotFoundException if the session
                does not exist.

    Returns:
        The created session

    Raises:
        ResourceNotFoundException: If the session does not exist and create is false
    """
    try:
        stmt = (
            select(models.Session)
            .where(models.Session.workspace_name == workspace_name)
            .where(models.Session.name == session.name)
        )

        result = await db.execute(stmt)

        honcho_session = result.scalar_one_or_none()

        # Check if session already exists
        if honcho_session is None:
            if not create:
                raise ResourceNotFoundException(
                    f"Session {session.name} not found in workspace {workspace_name}"
                )

            # Create honcho session

            honcho_session = models.Session(
                workspace_name=workspace_name,
                name=session.name,
                h_metadata=session.metadata,
                feature_flags=session.feature_flags,
            )
            db.add(honcho_session)
            # Flush to ensure session exists in DB before adding peers
            await db.flush()

        # Add all peers to session
        if session.peer_names:
            await get_or_create_peers(
                db,
                workspace_name=workspace_name,
                peers=[
                    schemas.PeerCreate(name=peer_name)
                    for peer_name in session.peer_names
                ],
            )
            await _add_peers_to_session(
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
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating session in workspace {workspace_name}: {str(e)}")
        raise


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

    if session.feature_flags is not None:
        honcho_session.feature_flags = session.feature_flags

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
    cutoff_message_id: Optional[str] = None,
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
) -> Select:
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
    stmt = (
        select(models.Peer)
        .join(models.SessionPeer, models.Peer.name == models.SessionPeer.peer_name)
        .where(models.SessionPeer.session_name == session_name)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.SessionPeer.left_at.is_(None))  # Only active peers
    )

    return stmt


async def get_session_peer_feature_flags(
    workspace_name: str,
    session_name: str,
) -> Select:
    """
    Get feature flags from both SessionPeer and Peer tables for active peers in a session.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        Select statement returning peer_name, peer_feature_flags, and session_peer_feature_flags
    """
    stmt = (
        select(
            models.Peer.name.label("peer_name"),
            models.Peer.feature_flags.label("peer_feature_flags"),
            models.SessionPeer.feature_flags.label("session_peer_feature_flags"),
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

    # Delete all existing session peers
    delete_stmt = delete(models.SessionPeer).where(
        models.SessionPeer.session_name == session_name
    )
    result = await db.execute(delete_stmt)

    # Get or create peers
    await get_or_create_peers(
        db,
        workspace_name=workspace_name,
        peers=[schemas.PeerCreate(name=peer_name) for peer_name in peer_names],
    )

    # Add new peers to session
    peers = await _add_peers_to_session(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        peer_names=peer_names,
    )

    await db.commit()
    return peers


async def _add_peers_to_session(
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

    # Use upsert to handle both new peers and rejoining peers
    stmt = pg_insert(models.SessionPeer).values(
        [
            {
                "session_name": session_name,
                "peer_name": peer_name,
                "workspace_name": workspace_name,
                "joined_at": func.now(),
                "left_at": None,
                "feature_flags": feature_flags.model_dump(),
            }
            for peer_name, feature_flags in peer_names.items()
        ]
    )

    # On conflict, update joined_at and clear left_at (rejoin scenario)
    stmt = stmt.on_conflict_do_update(
        index_elements=["session_name", "peer_name", "workspace_name"],
        set_={
            "joined_at": func.now(),
            "left_at": None,
        },
    )
    await db.execute(stmt)

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

    return schemas.SessionPeerConfig(**session_peer.feature_flags)


async def set_peer_config(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    peer_id: str,
    config: schemas.SessionPeerConfig,
) -> None:
    """
    Set the configuration for a peer in a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_id: Name of the peer
        config: Configuration for the peer

    Returns:
        True if the peer config was set successfully

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

    # Update peer config
    session_peer.feature_flags["observe_others"] = config.observe_others
    session_peer.feature_flags["observe_me"] = config.observe_me

    await db.commit()
    return


async def search(
    query: str,
    *,
    workspace_name: str,
    session_name: Optional[str] = None,
    peer_name: Optional[str] = None,
) -> Select:
    """
    Search across message content. If a session or peer is provided,
    the search will be scoped to that session or peer. Otherwise, it will
    search across all messages in the workspace.

    Args:
        query: Search query to match against message content
        workspace_name: Name of the workspace
        session_name: Optional name of the session
        peer_name: Optional name of the peer

    Returns:
        List of messages that match the search query
    """
    if session_name is not None:
        stmt = select(models.Message).where(
            models.Message.session_name == session_name,
            models.Message.workspace_name == workspace_name,
            models.Message.content.ilike(f"%{query}%"),
        )
    elif peer_name is not None:
        stmt = select(models.Message).where(
            models.Message.peer_name == peer_name,
            models.Message.workspace_name == workspace_name,
            models.Message.content.ilike(f"%{query}%"),
        )
    else:
        stmt = select(models.Message).where(
            models.Message.workspace_name == workspace_name,
            models.Message.content.ilike(f"%{query}%"),
        )

    return stmt


########################################################
# Message Methods
########################################################


async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    workspace_name: str,
    session_name: str,
) -> list[models.Message]:
    """
    Bulk create messages for a session while maintaining order.

    Args:
        db: Database session
        messages: List of messages to create
        workspace_name: Name of the workspace
        session_name: Name of the session to create messages in

    Returns:
        List of created message objects
    """
    try:
        # Get or create session with peers in messages list
        peers = {message.peer_name: schemas.SessionPeerConfig() for message in messages}
        await get_or_create_session(
            db,
            session=schemas.SessionCreate(
                name=session_name,
                peer_names=peers,
            ),
            workspace_name=workspace_name,
        )

        # Create list of message objects (this will trigger the before_insert event)
        message_objects = [
            models.Message(
                session_name=session_name,
                peer_name=message.peer_name,
                content=message.content,
                h_metadata=message.metadata or {},
                workspace_name=workspace_name,
            )
            for message in messages
        ]

        # Add all messages and commit
        db.add_all(message_objects)
        await db.commit()

        return message_objects
    except IntegrityError as e:
        await db.rollback()
        if "fk_messages_peer_name_peers" in str(e):
            raise ResourceNotFoundException(
                f"One or more peers not found in workspace '{workspace_name}'"
            ) from e
        if "fk_messages_session_name_sessions" in str(e):
            raise ResourceNotFoundException(
                f"Session '{session_name}' not found in workspace '{workspace_name}'"
            ) from e
        raise


async def create_messages_for_peer(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    workspace_name: str,
    peer_name: str,
) -> list[models.Message]:
    """
    Bulk create messages for a peer while maintaining order.
    Note that session_name for messages created this way will be None
    and peer_name will be the provided peer_name for each message,
    regardless of the peer_name in the individual message(s).

    Args:
        db: Database session
        messages: List of messages to create
        workspace_name: Name of the workspace
        peer_name: Name of the peer to create messages for

    Returns:
        List of created message objects

    Raises:
        ResourceNotFoundException: If the peer does not exist in the workspace
    """
    try:
        await get_or_create_peer(
            db, workspace_name=workspace_name, peer=schemas.PeerCreate(name=peer_name)
        )
        # Create list of message objects (this will trigger the before_insert event)
        message_objects = [
            models.Message(
                session_name=None,
                peer_name=peer_name,
                content=message.content,
                h_metadata=message.metadata or {},
                workspace_name=workspace_name,
            )
            for message in messages
        ]

        # Add all messages and commit
        db.add_all(message_objects)
        await db.commit()

        return message_objects
    except IntegrityError as e:
        await db.rollback()
        if "fk_messages_peer_name_peers" in str(e):
            raise ResourceNotFoundException(
                f"Peer '{peer_name}' not found in workspace '{workspace_name}'"
            ) from e
        raise


async def get_messages(
    workspace_name: str,
    session_name: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
    token_limit: Optional[int] = None,
    message_count_limit: Optional[int] = None,
) -> Select:
    """
    Get messages from a session. If token_limit is provided, the n most recent messages
    with token count adding up to the limit will be returned. If message_count_limit is provided,
    the n most recent messages will be returned. If both are provided, message_count_limit will be
    used.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session
        reverse: Whether to reverse the order of messages
        filter: Filter to apply to the messages
        token_limit: Maximum number of tokens to include in the messages
        message_count_limit: Maximum number of messages to include

    Returns:
        Select statement for the messages
    """
    # Base query with workspace and session filters
    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    # Add metadata filter if provided
    if filter is not None:
        base_conditions.append(models.Message.h_metadata.contains(filter))

    # Apply message count limit first (takes precedence over token limit)
    if message_count_limit is not None:
        stmt = select(models.Message).where(*base_conditions)
        # For message count limit, we want the most recent N messages
        # So we order by id desc to get most recent, then apply limit
        stmt = stmt.order_by(models.Message.id.desc()).limit(message_count_limit)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    elif token_limit is not None:
        # Apply token limit logic
        # Create a subquery that calculates running sum of tokens for most recent messages
        token_subquery = (
            select(
                models.Message.id,
                func.sum(models.Message.token_count)
                .over(order_by=models.Message.id.desc())
                .label("running_token_sum"),
            )
            .where(*base_conditions)
            .subquery()
        )

        # Select Message objects where running sum doesn't exceed token_limit
        stmt = (
            select(models.Message)
            .join(token_subquery, models.Message.id == token_subquery.c.id)
            .where(token_subquery.c.running_token_sum <= token_limit)
        )

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    else:
        # Default case - no limits applied
        stmt = select(models.Message).where(*base_conditions)
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    return stmt


async def get_messages_id_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str | None,
    peer_name: str | None,
    start_id: int = 0,
    end_id: Optional[int] = None,
) -> list[models.Message]:
    """
    Get messages from a session or peer by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is exclusive of the end_id message.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        peer_name: Name of the peer
        start_id: Primary key ID of the first message to return
        end_id: Primary key ID of the last message (exclusive)

    Returns:
        List of messages

    Raises:
        ValueError: If both session_name and peer_name are not provided
    """
    if end_id is not None and (start_id >= end_id or end_id <= 1):
        return []
    stmt = select(models.Message).where(
        models.Message.workspace_name == workspace_name,
    )
    if end_id:
        stmt = stmt.where(models.Message.id.between(start_id, end_id - 1))
    else:
        stmt = stmt.where(models.Message.id >= start_id)

    if session_name:
        stmt = stmt.where(models.Message.session_name == session_name)
    elif peer_name:
        stmt = stmt.where(models.Message.peer_name == peer_name).where(
            models.Message.session_name.is_(None)
        )
    else:
        raise ValueError("Either session_name or peer_name must be provided")
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_messages_for_peer(
    workspace_name: str,
    peer_name: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.peer_name == peer_name)
        .where(models.Message.session_name.is_(None))
    )

    if filter is not None:
        stmt = stmt.where(models.Message.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Message.id.desc())
    else:
        stmt = stmt.order_by(models.Message.id)

    return stmt


async def get_message(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> Optional[models.Message]:
    stmt = (
        select(models.Message)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> bool:
    honcho_message = await get_message(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        message_id=message_id,
    )
    if honcho_message is None:
        raise ValueError("Message not found or does not belong to user")
    if (
        message.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_message.h_metadata = message.metadata
    await db.commit()
    # await db.refresh(honcho_message)
    return honcho_message


########################################################
# collection methods
########################################################

# Should be very similar to the session methods


async def get_collection(
    db: AsyncSession, workspace_name: str, peer_name: str, collection_name: str
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
        .where(models.Collection.peer_name == peer_name)
        .where(models.Collection.name == collection_name)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(
            f"Collection with name '{collection_name}' not found for peer {peer_name}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to peer"
        )
    return collection


async def get_or_create_peer_protected_collection(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
) -> models.Collection:
    try:
        honcho_collection = await get_collection(
            db, workspace_name, peer_name, DEF_PROTECTED_COLLECTION_NAME
        )
        return honcho_collection
    except ResourceNotFoundException:
        honcho_collection = models.Collection(
            workspace_name=workspace_name,
            peer_name=peer_name,
            name=DEF_PROTECTED_COLLECTION_NAME,
        )
        db.add(honcho_collection)
        await db.commit()
        return honcho_collection


########################################################
# document methods
########################################################


async def query_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    query: str,
    filter: Optional[dict] = None,
    max_distance: Optional[float] = None,
    top_k: int = 5,
) -> Sequence[models.Document]:
    # Using async client with await
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    )
    embedding_query = response.data[0].embedding
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.peer_name == peer_name)
        .where(models.Document.collection_name == collection_name)
        # .limit(top_k)
    )
    if max_distance is not None:
        stmt = stmt.where(
            models.Document.embedding.cosine_distance(embedding_query) < max_distance
        )
    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))
    stmt = stmt.limit(top_k).order_by(
        models.Document.embedding.cosine_distance(embedding_query)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_document(
    db: AsyncSession,
    document: schemas.DocumentCreate,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    duplicate_threshold: Optional[float] = None,
) -> models.Document:
    """
    Embed text as a vector and create a document.

    Args:
        db: Database session
        document: Document creation schema
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection

    Returns:
        The created document

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """

    # This will raise ResourceNotFoundException if collection not found
    await get_collection(
        db,
        workspace_name=workspace_name,
        peer_name=peer_name,
        collection_name=collection_name,
    )

    # Using async client with await
    response = await openai_client.embeddings.create(
        input=document.content, model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    if duplicate_threshold is not None:
        # Check if there are duplicates within the threshold
        stmt = (
            select(models.Document)
            .where(models.Document.collection_name == collection_name)
            .where(
                models.Document.embedding.cosine_distance(embedding)
                < duplicate_threshold
            )
            .order_by(models.Document.embedding.cosine_distance(embedding))
            .limit(1)
        )
        result = await db.execute(stmt)
        duplicate = result.scalar_one_or_none()  # Get the closest match if any exist
        if duplicate is not None:
            logger.info(f"Duplicate found: {duplicate.content}. Ignoring new document.")
            return duplicate

    honcho_document = models.Document(
        workspace_name=workspace_name,
        peer_name=peer_name,
        collection_name=collection_name,
        content=document.content,
        h_metadata=document.metadata,
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    await db.refresh(honcho_document)
    return honcho_document


async def get_duplicate_documents(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    collection_name: str,
    content: str,
    similarity_threshold: float = 0.85,
) -> list[models.Document]:
    """Check if a document with similar content already exists in the collection.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        collection_name: Name of the collection
        content: Document content to check for duplicates
        similarity_threshold: Similarity threshold (0-1) for considering documents as duplicates

    Returns:
        List of documents that are similar to the provided content
    """
    # Get embedding for the content
    # Using async client with await
    response = await openai_client.embeddings.create(
        input=content, model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # Find documents with similar embeddings
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_name == workspace_name)
        .where(models.Document.peer_name == peer_name)
        .where(models.Document.collection_name == collection_name)
        .where(
            models.Document.embedding.cosine_distance(embedding)
            < (1 - similarity_threshold)
        )  # Convert similarity to distance
        .order_by(models.Document.embedding.cosine_distance(embedding))
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())  # Convert to list to match the return type
