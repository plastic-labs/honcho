from collections.abc import Sequence
from logging import getLogger
from typing import List, Optional

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
# app methods
########################################################


async def get_app(db: AsyncSession, app_id: str) -> models.App:
    """
    Get an app by its ID.

    Args:
        db: Database session
        app_id: Public ID of the app

    Returns:
        The app if found

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    stmt = select(models.App).where(models.App.public_id == app_id)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    if app is None:
        logger.warning(f"App with ID {app_id} not found")
        raise ResourceNotFoundException(f"App with ID {app_id} not found")
    return app


async def get_all_apps(
    db: AsyncSession,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """
    Get all apps.

    Args:
        db: Database session
        reverse: Whether to reverse the order of the apps
        filter: Filter the apps by a dictionary of metadata
    """
    stmt = select(models.App)
    if reverse:
        stmt = stmt.order_by(models.App.id.desc())
    else:
        stmt = stmt.order_by(models.App.id)
    if filter is not None:
        stmt = stmt.where(models.App.h_metadata.contains(filter))
    return stmt


async def get_app_by_name(db: AsyncSession, name: str) -> models.App:
    """
    Get an app by its name.

    Args:
        db: Database session
        name: Name of the app

    Returns:
        The app if found

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    stmt = select(models.App).where(models.App.name == name)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    if app is None:
        logger.warning(f"App with name '{name}' not found")
        raise ResourceNotFoundException(f"App with name '{name}' not found")
    return app


# def get_apps(db: AsyncSession) -> Sequence[models.App]:
#     return db.query(models.App).all()


async def create_app(db: AsyncSession, app: schemas.AppCreate) -> models.App:
    """
    Create a new app.

    Args:
        db: Database session
        app: App creation schema

    Returns:
        The created app

    Raises:
        ConflictException: If an app with the same name already exists
    """
    try:
        honcho_app = models.App(name=app.name, h_metadata=app.metadata)
        db.add(honcho_app)
        await db.commit()
        logger.info(f"App created successfully: {app.name}")
        return honcho_app
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError creating app with name '{app.name}': {str(e)}")
        raise ConflictException(f"App with name '{app.name}' already exists") from e


async def update_app(
    db: AsyncSession, app_id: str, app: schemas.AppUpdate
) -> models.App:
    """
    Update an app.

    Args:
        db: Database session
        app_id: Public ID of the app
        app: App update schema

    Returns:
        The updated app

    Raises:
        ResourceNotFoundException: If the app does not exist
    """
    try:
        honcho_app = await get_app(db, app_id)

        if app.name is not None:
            honcho_app.name = app.name
        if app.metadata is not None:
            honcho_app.h_metadata = app.metadata

        await db.commit()
        logger.info(f"App with ID {app_id} updated successfully")
        return honcho_app
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError updating app {app_id}: {str(e)}")
        raise ConflictException(
            "App update failed - unique constraint violation"
        ) from e


########################################################
# user methods
########################################################


async def create_user(
    db: AsyncSession, app_id: str, user: schemas.UserCreate
) -> models.User:
    """
    Create a new user.

    Args:
        db: Database session
        app_id: Public ID of the app
        user: User creation schema

    Returns:
        The created user

    Raises:
        ConflictException: If a user with the same name already exists in this app
    """
    try:
        honcho_user = models.User(
            app_id=app_id,
            name=user.name,
            h_metadata=user.metadata,
        )
        db.add(honcho_user)
        await db.commit()
        logger.info(f"User created successfully: {user.name} for app {app_id}")
        return honcho_user
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create user - integrity error: {str(e)}")
        raise ConflictException("User with this name already exists") from e


async def get_user(db: AsyncSession, app_id: str, user_id: str) -> models.User:
    """
    Get a user by app ID and user ID.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user

    Returns:
        The user if found

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning(f"User with ID '{user_id}' not found in app {app_id}")
        raise ResourceNotFoundException(f"User with ID '{user_id}' not found")
    return user


async def get_user_by_name(db: AsyncSession, app_id: str, name: str) -> models.User:
    """
    Get a user by app ID and name.

    Args:
        db: Database session
        app_id: Public ID of the app
        name: Name of the user

    Returns:
        The user if found

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.name == name)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning(f"User with name '{name}' not found in app {app_id}")
        raise ResourceNotFoundException(f"User with name '{name}' not found")
    return user


async def get_users(
    db: AsyncSession,
    app_id: str,
    reverse: bool = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = select(models.User).where(models.User.app_id == app_id)

    if filter is not None:
        stmt = stmt.where(models.User.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.User.id.desc())
    else:
        stmt = stmt.order_by(models.User.id)

    return stmt


async def update_user(
    db: AsyncSession, app_id: str, user_id: str, user: schemas.UserUpdate
) -> models.User:
    """
    Update a user.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        user: User update schema

    Returns:
        The updated user

    Raises:
        ResourceNotFoundException: If the user does not exist
        ValidationException: If the update data is invalid
        ConflictException: If the update violates a unique constraint
    """
    try:
        # get_user will raise ResourceNotFoundException if not found
        honcho_user = await get_user(db, app_id, user_id)

        if user.name is not None:
            honcho_user.name = user.name
        if user.metadata is not None:
            honcho_user.h_metadata = user.metadata

        await db.commit()
        logger.info(f"User {user_id} updated successfully")
        return honcho_user
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"User update failed due to integrity error: {str(e)}")
        raise ConflictException(
            "User update failed - unique constraint violation"
        ) from e


########################################################
# session methods
########################################################


async def get_session(
    db: AsyncSession,
    app_id: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> models.Session:
    """
    Get a session by ID for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        session_id: Public ID of the session
        user_id: Optional public ID of the user

    Returns:
        The session if found

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    stmt = (
        select(models.Session)
        .where(models.Session.app_id == app_id)
        .where(models.Session.public_id == session_id)
    )
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if session is None:
        logger.warning(f"Session with ID '{session_id}' not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")
    return session


async def get_sessions(
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    is_active: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Session)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
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
    app_id: str,
    user_id: str,
) -> models.Session:
    """
    Create a new session for a user.

    Args:
        db: Database session
        session: Session creation schema
        app_id: ID of the app
        user_id: ID of the user

    Returns:
        The created session

    Raises:
        ResourceNotFoundException: If the user does not exist
    """
    try:
        # This will raise ResourceNotFoundException if user not found
        _honcho_user = await get_user(db, app_id=app_id, user_id=user_id)

        honcho_session = models.Session(
            user_id=user_id,
            app_id=app_id,
            h_metadata=session.metadata,
        )
        db.add(honcho_session)
        await db.commit()
        logger.info(f"Session created successfully for user {user_id}")
        return honcho_session
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating session for user {user_id}: {str(e)}")
        raise


async def update_session(
    db: AsyncSession,
    session: schemas.SessionUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
) -> models.Session:
    """
    Update a session.

    Args:
        db: Database session
        session: Session update schema
        app_id: ID of the app
        user_id: ID of the user
        session_id: ID of the session

    Returns:
        The updated session

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")

    if (
        session.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_session.h_metadata = session.metadata

    await db.commit()
    logger.info(f"Session {session_id} updated successfully")
    return honcho_session


async def delete_session(
    db: AsyncSession, app_id: str, user_id: str, session_id: str
) -> bool:
    """
    Mark a session as inactive (soft delete).

    Args:
        db: Database session
        app_id: ID of the app
        user_id: ID of the user
        session_id: ID of the session

    Returns:
        True if the session was deleted successfully

    Raises:
        ResourceNotFoundException: If the session does not exist or doesn't belong to the user
    """
    stmt = (
        select(models.Session)
        .where(models.Session.public_id == session_id)
        .where(models.Session.user_id == user_id)
        .where(models.Session.app_id == app_id)
    )
    result = await db.execute(stmt)
    honcho_session = result.scalar_one_or_none()

    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for user {user_id}")
        raise ResourceNotFoundException("Session not found or does not belong to user")

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
        stmt = select(models.Metamessage).where(
            models.Metamessage.session_id == original_session_id
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
                    "metamessage_type": meta.metamessage_type,
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
    app_id: str,
    user_id: str,
    session_id: str,
) -> models.Message:
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_message = models.Message(
        session_id=session_id,
        is_user=message.is_user,
        content=message.content,
        h_metadata=message.metadata,
        user_id=user_id,
        app_id=app_id,
    )
    db.add(honcho_message)
    await db.commit()
    # await db.refresh(honcho_message, attribute_names=["id", "content", "h_metadata"])
    # await db.refresh(honcho_message)
    return honcho_message


async def create_messages(
    db: AsyncSession,
    messages: list[schemas.MessageCreate],
    app_id: str,
    user_id: str,
    session_id: str,
) -> list[models.Message]:
    """Bulk create messages for a session while maintaining order"""
    # Verify session exists and belongs to user
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")

    # Create list of message records
    message_records = [
        {
            "session_id": session_id,
            "is_user": message.is_user,
            "content": message.content,
            "h_metadata": message.metadata,
            "user_id": user_id,
            "app_id": app_id,
        }
        for message in messages
    ]

    # Bulk insert messages and return them in order
    stmt = insert(models.Message).returning(models.Message)
    result = await db.execute(stmt, message_records)
    await db.commit()

    return list(result.scalars().all())


async def get_messages(
    db: AsyncSession,
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
# metamessage methods
########################################################


async def create_metamessage(
    db: AsyncSession,
    user_id: str,
    metamessage: schemas.MetamessageCreate,
    app_id: str,
):
    # Validate user exists
    user = await get_user(db, app_id=app_id, user_id=user_id)
    if user is None:
        raise ResourceNotFoundException(f"User with ID '{user_id}' not found")

    # Initialize metamessage data
    metamessage_data = {
        "user_id": user_id,
        "app_id": app_id,
        "metamessage_type": metamessage.metamessage_type,
        "content": metamessage.content,
        "h_metadata": metamessage.metadata,
    }

    # Validate session_id if provided
    if metamessage.session_id is not None:
        session = await get_session(
            db,
            app_id=app_id,
            user_id=user_id,
            session_id=metamessage.session_id,
        )
        if session is None:
            raise ResourceNotFoundException(
                "Session not found or does not belong to user"
            )
        metamessage_data["session_id"] = metamessage.session_id

        # Validate message_id if provided
        if metamessage.message_id is not None:
            message = await get_message(
                db,
                app_id=app_id,
                session_id=metamessage.session_id,
                user_id=user_id,
                message_id=metamessage.message_id,
            )
            if message is None:
                raise ResourceNotFoundException(
                    "Message not found or does not belong to session"
                )
            metamessage_data["message_id"] = metamessage.message_id
    elif metamessage.message_id is not None:
        # If message_id provided but no session_id, that's an error
        raise ValidationException("Cannot specify message_id without session_id")

    # Create metamessage
    honcho_metamessage = models.Metamessage(**metamessage_data)
    db.add(honcho_metamessage)
    await db.commit()
    return honcho_metamessage


async def get_metamessages(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    metamessage_type: Optional[str] = None,
    filter: Optional[dict] = None,
    reverse: Optional[bool] = False,
) -> Select:
    # Base query starts with metamessage and user relationship
    stmt = (
        select(models.Metamessage)
        .where(models.Metamessage.app_id == app_id)
        .where(models.Metamessage.user_id == user_id)
    )

    # If session_id is provided, filter by it
    if session_id is not None:
        stmt = stmt.where(models.Metamessage.session_id == session_id)

    # If message_id is provided, filter by it
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)

    # Filter by metamessage_type if provided
    if metamessage_type is not None:
        stmt = stmt.where(models.Metamessage.metamessage_type == metamessage_type)

    # Apply metadata filter if provided
    if filter is not None:
        stmt = stmt.where(models.Metamessage.h_metadata.contains(filter))

    # Apply sort order
    if reverse:
        stmt = stmt.order_by(models.Metamessage.id.desc())
    else:
        stmt = stmt.order_by(models.Metamessage.id)

    return stmt


async def get_metamessage(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    metamessage_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Optional[models.Metamessage]:
    # Base query for metamessage by ID
    stmt = (
        select(models.Metamessage)
        .where(models.Metamessage.app_id == app_id)
        .where(models.Metamessage.user_id == user_id)
        .where(models.Metamessage.public_id == metamessage_id)
    )

    # Add optional filters
    if session_id is not None:
        stmt = stmt.where(models.Metamessage.session_id == session_id)
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_metamessage(
    db: AsyncSession,
    metamessage: schemas.MetamessageUpdate,
    app_id: str,
    user_id: str,
    metamessage_id: str,
) -> bool:
    # First retrieve the metamessage
    metamessage_obj = await get_metamessage(
        db, app_id=app_id, user_id=user_id, metamessage_id=metamessage_id
    )
    if metamessage_obj is None:
        raise ResourceNotFoundException(
            f"Metamessage with ID {metamessage_id} not found"
        )

    # Validate the consistency of relationships if they're being changed
    # If we're setting message_id, we must have a session_id
    if metamessage.message_id is not None and metamessage.session_id is None:
        # If updating message_id but not session_id, use the existing session_id
        metamessage.session_id = metamessage_obj.session_id
        if metamessage.session_id is None:
            raise ValidationException("Cannot specify message_id without session_id")

    # If we're updating session_id and message_id, validate they belong together
    if metamessage.session_id is not None and metamessage.message_id is not None:
        message = await get_message(
            db,
            app_id=app_id,
            session_id=metamessage.session_id,
            user_id=metamessage.user_id,
            message_id=metamessage.message_id,
        )
        if message is None:
            raise ResourceNotFoundException(
                "Message not found or doesn't belong to session"
            )

    # Update fields
    if metamessage.session_id is not None:
        metamessage_obj.session_id = metamessage.session_id

    if metamessage.message_id is not None:
        metamessage_obj.message_id = metamessage.message_id

    if metamessage.metadata is not None:
        metamessage_obj.h_metadata = metamessage.metadata

    if metamessage.metamessage_type is not None:
        metamessage_obj.metamessage_type = metamessage.metamessage_type

    await db.commit()
    return metamessage_obj


########################################################
# collection methods
########################################################

# Should be very similar to the session methods


async def get_collections(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """Get a distinct list of the names of collections associated with a user"""
    stmt = (
        select(models.Collection)
        .where(models.Collection.app_id == app_id)
        .where(models.Collection.user_id == user_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Collection.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Collection.id.desc())
    else:
        stmt = stmt.order_by(models.Collection.id)

    return stmt


async def get_collection_by_id(
    db: AsyncSession, app_id: str, user_id: str, collection_id: str
) -> models.Collection:
    """
    Get a collection by ID for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        collection_id: Public ID of the collection

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.app_id == app_id)
        .where(models.Collection.user_id == user_id)
        .where(models.Collection.public_id == collection_id)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(
            f"Collection with ID '{collection_id}' not found for user {user_id}"
        )
        raise ResourceNotFoundException(
            "Collection not found or does not belong to user"
        )
    return collection


async def get_collection_by_name(
    db: AsyncSession, app_id: str, user_id: str, name: str
) -> models.Collection:
    """
    Get a collection by name for a specific user and app.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        name: Name of the collection

    Returns:
        The collection if found

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    stmt = (
        select(models.Collection)
        .where(models.Collection.app_id == app_id)
        .where(models.Collection.user_id == user_id)
        .where(models.Collection.name == name)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    if collection is None:
        logger.warning(f"Collection with name '{name}' not found for user {user_id}")
        raise ResourceNotFoundException(f"Collection with name '{name}' not found")
    return collection


async def create_collection(
    db: AsyncSession,
    collection: schemas.CollectionCreate,
    app_id: str,
    user_id: str,
) -> models.Collection:
    """
    Create a new collection for a user.

    Args:
        db: Database session
        collection: Collection creation schema
        app_id: ID of the app
        user_id: ID of the user

    Returns:
        The created collection

    Raises:
        ConflictException: If a collection with the same name already exists for this user
        ValidationException: If the collection configuration is invalid
        ResourceNotFoundException: If the user does not exist
    """
    try:
        # This will raise ResourceNotFoundException if user not found
        await get_user(db, app_id=app_id, user_id=user_id)

        # Check for reserved names
        if collection.name == "honcho":
            logger.warning(
                f"Attempted to create collection with reserved name 'honcho' for user {user_id}"
            )
            raise ValidationException(
                "Invalid collection configuration - 'honcho' is a reserved name"
            )

        honcho_collection = models.Collection(
            user_id=user_id,
            app_id=app_id,
            name=collection.name,
            h_metadata=collection.metadata,
        )
        db.add(honcho_collection)
        await db.commit()
        logger.info(
            f"Collection '{collection.name}' created successfully for user {user_id}"
        )
        return honcho_collection
    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Failed to create collection - integrity error: {str(e)}")
        raise ConflictException(
            f"Collection with name '{collection.name}' already exists"
        ) from e


async def create_user_protected_collection(
    db: AsyncSession,
    app_id: str,
    user_id: str,
) -> models.Collection:
    honcho_collection = models.Collection(
        user_id=user_id,
        app_id=app_id,
        name=DEF_PROTECTED_COLLECTION_NAME,
    )
    try:
        db.add(honcho_collection)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Collection already exists") from None
    return honcho_collection


async def get_or_create_user_protected_collection(
    db: AsyncSession,
    app_id: str,
    user_id: str,
) -> models.Collection:
    try:
        honcho_collection = await get_collection_by_name(
            db, app_id, user_id, DEF_PROTECTED_COLLECTION_NAME
        )
        return honcho_collection
    except ResourceNotFoundException:
        honcho_collection = await create_user_protected_collection(db, app_id, user_id)
        return honcho_collection


async def update_collection(
    db: AsyncSession,
    collection: schemas.CollectionUpdate,
    app_id: str,
    user_id: str,
    collection_id: str,
) -> models.Collection:
    """
    Update a collection.

    Args:
        db: Database session
        collection: Collection update schema
        app_id: ID of the app
        user_id: ID of the user
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
            db, app_id=app_id, user_id=user_id, collection_id=collection_id
        )

        # Check for reserved names if name is being updated
        if collection.name == "honcho":
            logger.warning(
                f"Attempted to rename collection to reserved name 'honcho' for user {user_id}"
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
    db: AsyncSession, app_id: str, user_id: str, collection_id: str
) -> bool:
    """
    Delete a Collection and all documents associated with it. Takes advantage of
    the orm cascade feature.

    Args:
        db: Database session
        app_id: ID of the app
        user_id: ID of the user
        collection_id: ID of the collection

    Returns:
        True if the collection was deleted successfully

    Raises:
        ResourceNotFoundException: If the collection does not exist
    """
    try:
        # This will raise ResourceNotFoundException if not found
        honcho_collection = await get_collection_by_id(
            db, app_id=app_id, user_id=user_id, collection_id=collection_id
        )

        if honcho_collection.name == "honcho":
            logger.warning(
                f"Attempted to delete collection with reserved name 'honcho' for user {user_id}"
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
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Document)
        .where(models.Document.app_id == app_id)
        .where(models.Document.user_id == user_id)
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
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> models.Document:
    """
    Get a document by ID.

    Args:
        db: Database session
        app_id: Public ID of the app
        user_id: Public ID of the user
        collection_id: Public ID of the collection
        document_id: Public ID of the document

    Returns:
        The document if found

    Raises:
        ResourceNotFoundException: If the document does not exist
    """
    stmt = (
        select(models.Document)
        .where(models.Document.app_id == app_id)
        .where(models.Document.user_id == user_id)
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
    app_id: str,
    user_id: str,
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
        .where(models.Document.app_id == app_id)
        .where(models.Document.user_id == user_id)
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
    app_id: str,
    user_id: str,
    collection_id: str,
    duplicate_threshold: Optional[float] = None,
) -> models.Document:
    """
    Embed text as a vector and create a document.

    Args:
        db: Database session
        document: Document creation schema
        app_id: ID of the app
        user_id: ID of the user
        collection_id: ID of the collection

    Returns:
        The created document

    Raises:
        ResourceNotFoundException: If the collection does not exist
        ValidationException: If the document data is invalid
    """

    # This will raise ResourceNotFoundException if collection not found
    collection = await get_collection_by_id(
        db, app_id=app_id, collection_id=collection_id, user_id=user_id
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
        app_id=app_id,
        user_id=user_id,
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
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    honcho_document = await get_document(
        db,
        app_id=app_id,
        collection_id=collection_id,
        user_id=user_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise ValueError("Session not found or does not belong to user")
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
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    honcho_collection = await get_collection_by_id(
        db, app_id=app_id, collection_id=collection_id, user_id=user_id
    )
    if honcho_collection is None:
        raise ResourceNotFoundException("Collection or Document not found")
    if honcho_collection.name == "honcho":
        logger.warning(
            f"Attempted to delete collection with reserved name 'honcho' for user {user_id}"
        )
        raise ValidationException(
            "Cannot delete collection with reserved name 'honcho'"
        )
    stmt = (
        select(models.Document)
        .where(models.Document.app_id == app_id)
        .where(models.Document.user_id == user_id)
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
    app_id: str,
    user_id: str,
    collection_id: str,
    content: str,
    similarity_threshold: float = 0.85,
) -> List[models.Document]:
    """Check if a document with similar content already exists in the collection.

    Args:
        db: Database session
        app_id: Application ID
        user_id: User ID
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
        .where(models.Document.collection_id == collection_id)
        .where(
            models.Document.embedding.cosine_distance(embedding)
            < (1 - similarity_threshold)
        )  # Convert similarity to distance
        .order_by(models.Document.embedding.cosine_distance(embedding))
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())  # Convert to list to match the return type
