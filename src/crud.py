from collections.abc import Sequence
from logging import getLogger
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from sqlalchemy import Select, cast, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func
from sqlalchemy.types import BigInteger

from . import models, schemas
from .exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)

load_dotenv(override=True)

openai_client = AsyncOpenAI()

logger = getLogger(__name__)

DEF_PROTECTED_COLLECTION_NAME = "honcho"

########################################################
# workspace methods
########################################################


async def get_workspace(db: AsyncSession, workspace_id: str) -> models.Workspace:
    """
    Get a workspace by its ID.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace

    Returns:
        The workspace if found

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    stmt = select(models.Workspace).where(models.Workspace.public_id == workspace_id)
    result = await db.execute(stmt)
    workspace = result.scalar_one_or_none()
    if workspace is None:
        logger.warning(f"Workspace with ID {workspace_id} not found")
        raise ResourceNotFoundException(f"Workspace with ID {workspace_id} not found")
    return workspace


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


async def get_workspace_by_name(db: AsyncSession, name: str) -> models.Workspace:
    """
    Get a workspace by its name.

    Args:
        db: Database session
        name: Name of the workspace

    Returns:
        The workspace if found

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    stmt = select(models.Workspace).where(models.Workspace.name == name)
    result = await db.execute(stmt)
    workspace = result.scalar_one_or_none()
    if workspace is None:
        logger.warning(f"Workspace with name '{name}' not found")
        raise ResourceNotFoundException(f"Workspace with name '{name}' not found")
    return workspace


# def get_apps(db: AsyncSession) -> Sequence[models.App]:
#     return db.query(models.App).all()


async def create_workspace(db: AsyncSession, workspace: schemas.WorkspaceCreate) -> models.Workspace:
    """
    Create a new workspace.

    Args:
        db: Database session
        workspace: Workspace creation schema

    Returns:
        The created workspace

    Raises:
        ConflictException: If a workspace with the same name already exists
    """
    try:
        honcho_workspace = models.Workspace(name=workspace.name, h_metadata=workspace.metadata)
        db.add(honcho_workspace)
        await db.commit()
        logger.info(f"Workspace created successfully: {workspace.name}")
        return honcho_workspace
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError creating workspace with name '{workspace.name}': {str(e)}")
        raise ConflictException(f"Workspace with name '{workspace.name}' already exists") from e


async def update_workspace(
    db: AsyncSession, workspace_id: str, workspace: schemas.WorkspaceUpdate
) -> models.Workspace:
    """
    Update a workspace.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        workspace: Workspace update schema

    Returns:
        The updated workspace

    Raises:
        ResourceNotFoundException: If the workspace does not exist
    """
    try:
        honcho_workspace = await get_workspace(db, workspace_id)

        if workspace.name is not None:
            honcho_workspace.name = workspace.name
        if workspace.metadata is not None:
            honcho_workspace.h_metadata = workspace.metadata

        await db.commit()
        logger.info(f"Workspace with ID {workspace_id} updated successfully")
        return honcho_workspace
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError updating workspace {workspace_id}: {str(e)}")
        raise ConflictException(
            "Workspace update failed - unique constraint violation"
        ) from e


########################################################
# peer methods
########################################################


async def create_peer(
    db: AsyncSession, workspace_id: str, peer: schemas.PeerCreate
) -> models.Peer:
    """
    Create a new peer.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer: Peer creation schema

    Returns:
        The created peer

    Raises:
        ConflictException: If a peer with the same name already exists in this workspace
    """
    try:
        honcho_peer = models.Peer(
            workspace_id=workspace_id,
            name=peer.name,
            h_metadata=peer.metadata,
        )
        db.add(honcho_peer)
        await db.commit()
        logger.info(f"Peer created successfully: {peer.name} for workspace {workspace_id}")
        return honcho_peer
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create peer - integrity error: {str(e)}")
        raise ConflictException("Peer with this name already exists") from e


async def get_peer(db: AsyncSession, workspace_id: str, peer_id: str) -> models.Peer:
    """
    Get a peer by workspace ID and peer ID.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer

    Returns:
        The peer if found

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_id == workspace_id)
        .where(models.Peer.public_id == peer_id)
    )
    result = await db.execute(stmt)
    peer = result.scalar_one_or_none()
    if peer is None:
        logger.warning(f"Peer with ID '{peer_id}' not found in workspace {workspace_id}")
        raise ResourceNotFoundException(f"Peer with ID '{peer_id}' not found")
    return peer


async def get_peer_by_name(db: AsyncSession, workspace_id: str, name: str) -> models.Peer:
    """
    Get a peer by workspace ID and name.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        name: Name of the peer

    Returns:
        The peer if found

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_id == workspace_id)
        .where(models.Peer.name == name)
    )
    result = await db.execute(stmt)
    peer = result.scalar_one_or_none()
    if peer is None:
        logger.warning(f"Peer with name '{name}' not found in workspace {workspace_id}")
        raise ResourceNotFoundException(f"Peer with name '{name}' not found")
    return peer


async def get_peers(
    workspace_id: str,
    reverse: bool = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = select(models.Peer).where(models.Peer.workspace_id == workspace_id)

    if filter is not None:
        stmt = stmt.where(models.Peer.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Peer.id.desc())
    else:
        stmt = stmt.order_by(models.Peer.id)

    return stmt


async def update_peer(
    db: AsyncSession, workspace_id: str, peer_id: str, peer: schemas.PeerUpdate
) -> models.Peer:
    """
    Update a peer.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer
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
        honcho_peer = await get_peer(db, workspace_id, peer_id)

        if peer.name is not None:
            honcho_peer.name = peer.name
        if peer.metadata is not None:
            honcho_peer.h_metadata = peer.metadata

        await db.commit()
        logger.info(f"Peer {peer_id} updated successfully")
        return honcho_peer
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Peer update failed due to integrity error: {str(e)}")
        raise ConflictException(
            "Peer update failed - unique constraint violation"
        ) from e


async def get_peer_sessions(
    workspace_id: str,
    peer_id: str,
    reverse: bool = False,
    is_active: Optional[bool] = None,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all sessions for a peer through the session_peers relationship.

    Args:
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer
        reverse: Whether to reverse the order of the sessions
        is_active: Filter by active status (True/False/None for all)
        filter: Filter sessions by metadata

    Returns:
        SQLAlchemy Select statement
    """
    stmt = (
        select(models.Session)
        .join(models.SessionPeer, models.Session.public_id == models.SessionPeer.session_public_id)
        .where(models.SessionPeer.peer_public_id == peer_id)
        .where(models.Session.workspace_id == workspace_id)
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


async def get_session(
    db: AsyncSession,
    workspace_id: str,
    session_id: str,
    peer_id: Optional[str] = None,
) -> models.Session:
    """
    Get a session by ID for a specific workspace and optionally peer.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        session_id: Public ID of the session
        peer_id: Optional public ID of the peer (to check if peer is in session)

    Returns:
        The session if found

    Raises:
        ResourceNotFoundException: If the session does not exist or peer is not in session
    """
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_id == workspace_id)
        .where(models.Session.public_id == session_id)
    )
    if peer_id is not None:
        # Join with session_peers to check if peer is in session
        stmt = stmt.join(models.SessionPeer, models.Session.public_id == models.SessionPeer.session_public_id)
        stmt = stmt.where(models.SessionPeer.peer_public_id == peer_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if session is None:
        logger.warning(f"Session with ID '{session_id}' not found for peer {peer_id}")
        raise ResourceNotFoundException("Session not found or peer not in session")
    return session


async def get_sessions(
    workspace_id: str,
    reverse: Optional[bool] = False,
    is_active: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all sessions in a workspace.
    """
    stmt = (
        select(models.Session)
        .where(models.Session.workspace_id == workspace_id)
    )

    if is_active:
        stmt = stmt.where(models.Session.is_active.is_(True))

    if filter is not None:
        stmt = stmt.where(models.Session.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Session.id.desc())
    else:
        stmt = stmt.order_by(models.Session.id)

    return stmt


async def create_session(
    db: AsyncSession,
    session: schemas.SessionCreate,
    workspace_id: str,
    peer_ids: list[str] = None,
) -> models.Session:
    """
    Create a new session in a workspace with specified peers.

    Args:
        db: Database session
        session: Session creation schema
        workspace_id: ID of the workspace
        peer_ids: List of peer IDs to add to the session

    Returns:
        The created session

    Raises:
        ResourceNotFoundException: If any peer does not exist
    """
    try:
        # Verify all peers exist if provided
        if peer_ids:
            for peer_id in peer_ids:
                await get_peer(db, workspace_id=workspace_id, peer_id=peer_id)

        honcho_session = models.Session(
            workspace_id=workspace_id,
            h_metadata=session.metadata,
        )
        db.add(honcho_session)
        await db.flush()  # Get the session ID

        # Add peers to session if provided
        if peer_ids:
            for peer_id in peer_ids:
                session_peer = models.SessionPeer(
                    session_public_id=honcho_session.public_id,
                    peer_public_id=peer_id
                )
                db.add(session_peer)

        await db.commit()
        logger.info(f"Session created successfully in workspace {workspace_id} with {len(peer_ids or [])} peers")
        return honcho_session
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating session in workspace {workspace_id}: {str(e)}")
        raise


async def update_session(
    db: AsyncSession,
    session: schemas.SessionUpdate,
    workspace_id: str,
    session_id: str,
    peer_id: Optional[str] = None,
) -> models.Session:
    """
    Update a session.

    Args:
        db: Database session
        session: Session update schema
        workspace_id: ID of the workspace
        session_id: ID of the session
        peer_id: Optional ID of the peer (to verify peer is in session)

    Returns:
        The updated session

    Raises:
        ResourceNotFoundException: If the session does not exist or peer is not in session
    """
    honcho_session = await get_session(
        db, workspace_id=workspace_id, session_id=session_id, peer_id=peer_id
    )
    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for peer {peer_id}")
        raise ResourceNotFoundException("Session not found or peer not in session")

    if (
        session.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_session.h_metadata = session.metadata

    await db.commit()
    logger.info(f"Session {session_id} updated successfully")
    return honcho_session


async def delete_session(
    db: AsyncSession, workspace_id: str, session_id: str
) -> bool:
    """
    Mark a session as inactive (soft delete).

    Args:
        db: Database session
        workspace_id: ID of the workspace
        session_id: ID of the session

    Returns:
        True if the session was deleted successfully

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    stmt = (
        select(models.Session)
        .where(models.Session.public_id == session_id)
        .where(models.Session.workspace_id == workspace_id)
    )
    result = await db.execute(stmt)
    honcho_session = result.scalar_one_or_none()

    if honcho_session is None:
        logger.warning(f"Session {session_id} not found in workspace {workspace_id}")
        raise ResourceNotFoundException("Session not found")

    honcho_session.is_active = False
    await db.commit()
    logger.info(f"Session {session_id} marked as inactive")
    return True


async def clone_session(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    original_session_id: str,
    cutoff_message_id: Optional[str] = None,
    deep_copy: bool = True,
) -> models.Session:
    """
    Clone a session and its messages. If cutoff_message_id is provided,
    only clone messages up to and including that message.

    Args:
        db: SQLAlchemy session
        app_id: ID of the app the target session is in
        user_id: ID of the user the target session belongs to
        original_session_id: ID of the session to clone
        cutoff_message_id: Optional ID of the last message to include in the clone

    Returns:
        The newly created session
    """
    # Get the original session
    stmt = (
        select(models.Session)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Session.public_id == original_session_id)
    )
    original_session = await db.scalar(stmt)
    if not original_session:
        raise ValueError("Original session not found")

    # If cutoff_message_id is provided, verify it belongs to the session
    cutoff_message = None
    if cutoff_message_id is not None:
        stmt = select(models.Message).where(
            models.Message.public_id == cutoff_message_id,
            models.Message.session_id == original_session_id,
        )
        cutoff_message = await db.scalar(stmt)
        if not cutoff_message:
            raise ValueError(
                "Message not found or doesn't belong to the specified session"
            )

    # Create new session
    new_session = models.Session(
        user_id=original_session.user_id,
        app_id=original_session.app_id,
        h_metadata=original_session.h_metadata,
    )
    db.add(new_session)
    await db.flush()  # Flush to get the new session ID

    # Build query for messages to clone
    stmt = select(models.Message).where(
        models.Message.session_id == original_session_id
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
            "session_id": new_session.public_id,
            "content": message.content,
            "is_user": message.is_user,
            "h_metadata": message.h_metadata,
            "app_id": original_session.app_id,
            "user_id": original_session.user_id,
        }
        for message in messages_to_clone
    ]

    stmt = insert(models.Message).returning(models.Message.public_id)
    result = await db.execute(stmt, new_messages)
    new_message_ids = result.scalars().all()

    # Create mapping of old to new message IDs
    message_id_map = dict(
        zip([message.public_id for message in messages_to_clone], new_message_ids)
    )

    # Handle metamessages if deep copy is requested
    if deep_copy:
        # Fetch all metamessages tied to the session in a single query
        stmt = (
            select(models.Metamessage)
            .where(models.Metamessage.session_id == original_session_id)
            .order_by(models.Metamessage.id)  # Explicit ordering by id
        )
        if cutoff_message_id is not None and cutoff_message is not None:
            # Only get metamessages related to messages we're cloning
            message_ids = [message.public_id for message in messages_to_clone]
            stmt = stmt.where(
                (models.Metamessage.message_id.is_(None))
                | (models.Metamessage.message_id.in_(message_ids))
            )

        metamessages_result = await db.scalars(stmt)
        metamessages = metamessages_result.all()

        if metamessages:
            # Prepare bulk insert data for metamessages
            new_metamessages = []

            for meta in metamessages:
                # Base metamessage data
                meta_data = {
                    "user_id": meta.user_id,  # Preserve original user
                    "session_id": new_session.public_id,
                    "label": meta.label,
                    "content": meta.content,
                    "h_metadata": meta.h_metadata,
                    "app_id": original_session.app_id,
                }

                # If the metamessage was tied to a message, tie it to the corresponding new message
                if meta.message_id is not None and meta.message_id in message_id_map:
                    meta_data["message_id"] = message_id_map[meta.message_id]

                new_metamessages.append(meta_data)

            # Bulk insert metamessages using modern insert syntax
            if new_metamessages:
                stmt = insert(models.Metamessage)
                await db.execute(stmt, new_metamessages)

    await db.commit()

    return new_session


########################################################
# Message Methods
########################################################


async def create_message(
    db: AsyncSession,
    message: schemas.MessageCreate,
    workspace_id: str,
    session_id: str,
) -> models.Message:
    honcho_session = await get_session(
        db, workspace_id=workspace_id, session_id=session_id
    )
    if honcho_session is None:
        raise ValueError("Session not found")

    # Check if sender is already in the session
    stmt = (
        select(models.SessionPeer)
        .where(models.SessionPeer.session_public_id == session_id)
        .where(models.SessionPeer.peer_public_id == message.sender_id)
    )
    result = await db.execute(stmt)
    session_peer = result.scalar_one_or_none()
    
    if session_peer is None:
        # Sender is not in session - verify they are a valid peer in this workspace
        try:
            await get_peer(db, workspace_id=workspace_id, peer_id=message.sender_id)
            # Valid peer - add them to the session
            session_peer = models.SessionPeer(
                session_public_id=session_id,
                peer_public_id=message.sender_id
            )
            db.add(session_peer)
            logger.info(f"Added peer {message.sender_id} to session {session_id}")
        except ResourceNotFoundException:
            raise ValueError("Sender is not a valid peer in this workspace")

    honcho_message = models.Message(
        session_id=session_id,
        sender_id=message.sender_id,
        content=message.content,
        h_metadata=message.metadata,
        workspace_id=workspace_id,
    )
    db.add(honcho_message)
    await db.commit()
    # await db.refresh(honcho_message, attribute_names=["id", "content", "h_metadata"])
    # await db.refresh(honcho_message)
    return honcho_message


async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    workspace_id: str,
    session_id: str,
) -> list[models.Message]:
    """Bulk create messages for a session while maintaining order"""
    # Verify session exists
    honcho_session = await get_session(
        db, workspace_id=workspace_id, session_id=session_id
    )
    if honcho_session is None:
        raise ValueError("Session not found")

    # Get unique sender IDs and ensure they're all in the session
    sender_ids = list(set(message.sender_id for message in messages))
    
    # Check which senders are already in the session
    stmt = (
        select(models.SessionPeer.peer_public_id)
        .where(models.SessionPeer.session_public_id == session_id)
        .where(models.SessionPeer.peer_public_id.in_(sender_ids))
    )
    result = await db.execute(stmt)
    existing_peer_ids = set(result.scalars().all())
    
    # Add any missing peers to the session
    missing_peer_ids = set(sender_ids) - existing_peer_ids
    for peer_id in missing_peer_ids:
        # Verify they are valid peers in this workspace
        try:
            await get_peer(db, workspace_id=workspace_id, peer_id=peer_id)
            # Valid peer - add them to the session
            session_peer = models.SessionPeer(
                session_public_id=session_id,
                peer_public_id=peer_id
            )
            db.add(session_peer)
            logger.info(f"Added peer {peer_id} to session {session_id}")
        except ResourceNotFoundException:
            raise ValueError(f"Sender {peer_id} is not a valid peer in this workspace")

    # Create list of message records
    message_records = [
        {
            "session_id": session_id,
            "sender_id": message.sender_id,
            "content": message.content,
            "h_metadata": message.metadata,
            "workspace_id": workspace_id,
        }
        for message in messages
    ]

    # Bulk insert messages and return them in order
    stmt = insert(models.Message).returning(models.Message)
    result = await db.execute(stmt, message_records)
    await db.commit()

    return list(result.scalars().all())


async def get_messages(
    app_id: str,
    user_id: str,
    session_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Message)
        .where(models.Message.app_id == app_id)
        .where(models.Message.user_id == user_id)
        .where(models.Message.session_id == session_id)
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
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
) -> Optional[models.Message]:
    stmt = (
        select(models.Message)
        .where(models.Message.app_id == app_id)
        .where(models.Message.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
) -> bool:
    honcho_message = await get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
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


async def get_collections(
    workspace_id: str,
    peer_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """Get a distinct list of the names of collections associated with a peer"""
    stmt = (
        select(models.Collection)
        .where(models.Collection.workspace_id == workspace_id)
        .where(models.Collection.peer_id == peer_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Collection.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Collection.id.desc())
    else:
        stmt = stmt.order_by(models.Collection.id)

    return stmt


async def get_collection_by_id(
    db: AsyncSession, workspace_id: str, peer_id: str, collection_id: str
) -> models.Collection:
    """
    Get a collection by ID for a specific peer and workspace.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer
        collection_id: Public ID of the collection

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.workspace_id == workspace_id)
        .where(models.Collection.peer_id == peer_id)
        .where(models.Collection.public_id == collection_id)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(
            f"Collection with ID '{collection_id}' not found for peer {peer_id}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to peer"
        )
    return collection


async def get_collection_by_name(
    db: AsyncSession, workspace_id: str, peer_id: str, name: str
) -> models.Collection:
    """
    Get a collection by name for a specific peer and workspace.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer
        name: Name of the collection

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.workspace_id == workspace_id)
        .where(models.Collection.peer_id == peer_id)
        .where(models.Collection.name == name)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(f"Collection with name '{name}' not found for peer {peer_id}")
        raise ResourceNotFoundException(f"Collection with name '{name}' not found")
    return collection


async def create_collection(
    db: AsyncSession,
    collection: schemas.CollectionCreate,
    workspace_id: str,
    peer_id: str,
) -> models.Collection:
    """
    Create a new collection for a peer.

    Args:
        db: Database session
        collection: Collection creation schema
        workspace_id: ID of the workspace
        peer_id: ID of the peer

    Returns:
        The created collection

    Raises:
        ConflictException: If a collection with the same name already exists for this peer
        ValidationException: If the collection configuration is invalid
        ResourceNotFoundException: If the peer does not exist
    """
    try:
        # This will raise ResourceNotFoundException if peer not found
        await get_peer(db, workspace_id=workspace_id, peer_id=peer_id)

        # Check for reserved names
        if collection.name == "honcho":
            logger.warning(
                f"Attempted to create collection with reserved name 'honcho' for peer {peer_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        honcho_collection = models.Collection(
            peer_id=peer_id,
            workspace_id=workspace_id,
            name=collection.name,
            h_metadata=collection.metadata,
        )
        db.add(honcho_collection)
        await db.commit()
        logger.info(
            f"Collection '{collection.name}' created successfully for peer {peer_id}"
        )
        return honcho_collection
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create collection - integrity error: {str(e)}")
        raise ConflictException(
            f"Collection with name '{collection.name}' already exists"
        ) from e


async def create_peer_protected_collection(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
) -> models.Collection:
    honcho_collection = models.Collection(
        peer_id=peer_id,
        workspace_id=workspace_id,
        name=DEF_PROTECTED_COLLECTION_NAME,
    )
    try:
        # This will raise ResourceNotFoundException if peer not found
        await get_peer(db, workspace_id=workspace_id, peer_id=peer_id)

        db.add(honcho_collection)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Collection already exists") from None
    return honcho_collection


async def get_or_create_peer_protected_collection(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
) -> models.Collection:
    try:
        honcho_collection = await get_collection_by_name(
            db, workspace_id, peer_id, DEF_PROTECTED_COLLECTION_NAME
        )
        return honcho_collection
    except ResourceNotFoundException:
        honcho_collection = await create_peer_protected_collection(db, workspace_id, peer_id)
        return honcho_collection


async def update_collection(
    db: AsyncSession,
    collection: schemas.CollectionUpdate,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
) -> models.Collection:
    """
    Update a collection.

    Args:
        db: Database session
        collection: Collection update schema
        workspace_id: ID of the workspace
        peer_id: ID of the peer
        collection_id: ID of the collection

    Returns:
        The updated collection

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the update data is invalid
        ConflictException: If the update violates a unique constraint
    """
    try:
        # Validate input
        if collection.name is None and collection.metadata is None:
            logger.warning(
                f"Collection update attempted with no fields provided for collection {collection_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - at least one field must be provided"
            )

        # This will raise ResourceNotFoundException if not found
        honcho_collection = await get_collection_by_id(
            db, workspace_id=workspace_id, peer_id=peer_id, collection_id=collection_id
        )

        # Check for reserved names if name is being updated
        if collection.name == "honcho":
            logger.warning(
                f"Attempted to rename collection to reserved name 'honcho' for peer {peer_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        if collection.metadata is not None:
            honcho_collection.h_metadata = collection.metadata

        if collection.name is not None:
            honcho_collection.name = collection.name

        await db.commit()
        logger.info(f"Collection {collection_id} updated successfully")
        return honcho_collection
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Collection update failed due to integrity error: {str(e)}")
        raise ConflictException("Collection update failed - name already in use") from e


async def delete_collection(
    db: AsyncSession, workspace_id: str, peer_id: str, collection_id: str
) -> bool:
    """
    Delete a Collection and all documents associated with it. Takes advantage of
    the orm cascade feature.

    Args:
        db: Database session
        workspace_id: ID of the workspace
        peer_id: ID of the peer
        collection_id: ID of the collection

    Returns:
        True if the collection was deleted successfully

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    try:
        # This will raise ResourceNotFoundException if not found
        honcho_collection = await get_collection_by_id(
            db, workspace_id=workspace_id, peer_id=peer_id, collection_id=collection_id
        )

        if honcho_collection.name == "honcho":
            logger.warning(
                f"Attempted to delete collection with reserved name 'honcho' for peer {peer_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        await db.delete(honcho_collection)
        await db.commit()
        logger.info(f"Collection {collection_id} deleted successfully")
        return True
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting collection {collection_id}: {str(e)}")
        raise


########################################################
# document methods
########################################################

# Should be similar to the messages methods outside of query


async def get_documents(
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_id == workspace_id)
        .where(models.Document.peer_id == peer_id)
        .where(models.Document.collection_id == collection_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Document.id.desc())
    else:
        stmt = stmt.order_by(models.Document.id)

    return stmt


async def get_document(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    document_id: str,
) -> models.Document:
    """
    Get a document by ID.

    Args:
        db: Database session
        workspace_id: Public ID of the workspace
        peer_id: Public ID of the peer
        collection_id: Public ID of the collection
        document_id: Public ID of the document

    Returns:
        The document if found

    Raises:
        ResourceNotFoundException: If the document does not exist
    """
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_id == workspace_id)
        .where(models.Document.peer_id == peer_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )

    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if document is None:
        logger.warning(
            f"Document with ID '{document_id}' not found in collection {collection_id}"
        )
        raise ResourceNotFoundException(f"Document with ID '{document_id}' not found")
    return document


async def query_documents(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
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
        .where(models.Document.workspace_id == workspace_id)
        .where(models.Document.peer_id == peer_id)
        .where(models.Document.collection_id == collection_id)
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
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    duplicate_threshold: Optional[float] = None,
) -> models.Document:
    """
    Embed text as a vector and create a document.

    Args:
        db: Database session
        document: Document creation schema
        workspace_id: ID of the workspace
        peer_id: ID of the peer
        collection_id: ID of the collection

    Returns:
        The created document

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """

    # This will raise ResourceNotFoundException if collection not found
    await get_collection_by_id(
        db, workspace_id=workspace_id, collection_id=collection_id, peer_id=peer_id
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
            .where(models.Document.collection_id == collection_id)
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
        workspace_id=workspace_id,
        peer_id=peer_id,
        collection_id=collection_id,
        content=document.content,
        h_metadata=document.metadata,
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    await db.refresh(honcho_document)
    return honcho_document


async def update_document(
    db: AsyncSession,
    document: schemas.DocumentUpdate,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    honcho_document = await get_document(
        db,
        workspace_id=workspace_id,
        collection_id=collection_id,
        peer_id=peer_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise ValueError("Document not found or does not belong to peer")
    if document.content is not None:
        honcho_document.content = document.content
        # Using async client with await
        response = await openai_client.embeddings.create(
            input=document.content, model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        honcho_document.embedding = embedding
        honcho_document.created_at = func.now()

    if document.metadata is not None:
        honcho_document.h_metadata = document.metadata
    await db.commit()
    return honcho_document


async def delete_document(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    honcho_collection = await get_collection_by_id(
        db, workspace_id=workspace_id, collection_id=collection_id, peer_id=peer_id
    )
    if honcho_collection is None:
        raise ResourceNotFoundException("Collection or Document not found")
    if honcho_collection.name == "honcho":
        logger.warning(
            f"Attempted to delete collection with reserved name 'honcho' for peer {peer_id}"
        )
        raise ValidationException(
            "Cannot delete collection with reserved name 'honcho'"
        )
    stmt = (
        select(models.Document)
        .where(models.Document.workspace_id == workspace_id)
        .where(models.Document.peer_id == peer_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if document is None:
        return False
    await db.delete(document)
    await db.commit()
    return True


async def get_duplicate_documents(
    db: AsyncSession,
    workspace_id: str,
    peer_id: str,
    collection_id: str,
    content: str,
    similarity_threshold: float = 0.85,
) -> list[models.Document]:
    """Check if a document with similar content already exists in the collection.

    Args:
        db: Database session
        workspace_id: Workspace ID
        peer_id: Peer ID
        collection_id: Collection ID
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
        .where(models.Document.workspace_id == workspace_id)
        .where(models.Document.peer_id == peer_id)
        .where(models.Document.collection_id == collection_id)
        .where(
            models.Document.embedding.cosine_distance(embedding)
            < (1 - similarity_threshold)
        )  # Convert similarity to distance
        .order_by(models.Document.embedding.cosine_distance(embedding))
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())  # Convert to list to match the return type
