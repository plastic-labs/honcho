from logging import getLogger
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import ColumnElement, Select, and_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from src import models, schemas
from src.config import settings
from src.embedding_client import embedding_client
from src.utils.filter import apply_filter
from src.vector_store import VectorRecord, get_vector_store

from .session import get_or_create_session

logger = getLogger(__name__)


def _apply_token_limit(
    base_conditions: list[ColumnElement[Any]], token_limit: int
) -> Select[tuple[models.Message]]:
    """
    Helper function to apply token limit logic to a message query.

    Creates a subquery that calculates running sum of tokens for most recent messages
    and returns a select statement that joins with this subquery to limit results
    based on token count.

    Args:
        base_conditions: List of conditions to apply to the base query
        token_limit: Maximum number of tokens to include in the messages

    Returns:
        Select statement with token limit applied
    """
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
    return (
        select(models.Message)
        .join(token_subquery, models.Message.id == token_subquery.c.id)
        .where(token_subquery.c.running_token_sum <= token_limit)
    )


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
    # Get or create session with peers in messages list
    peers = {message.peer_name: schemas.SessionPeerConfig() for message in messages}
    await get_or_create_session(
        db,
        session=schemas.SessionCreate(name=session_name, peers=peers),
        workspace_name=workspace_name,
    )

    await db.execute(text("SET LOCAL lock_timeout = '5s'"))
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock(hashtext(:workspace_name), hashtext(:session_name))"
        ),
        {"workspace_name": workspace_name, "session_name": session_name},
    )

    # Get the last sequence number on a session - uses (workspace_name, session_name, seq_in_session) index
    last_seq = (
        await db.scalar(
            select(models.Message.seq_in_session)
            .where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
            )
            .order_by(models.Message.seq_in_session.desc())
            .limit(1)
        )
        or 0
    )

    # Create list of message objects (this will trigger the before_insert event)
    message_objects: list[models.Message] = []
    for offset, message in enumerate(messages, start=1):
        message_seq_in_session = last_seq + offset
        message_obj = models.Message(
            session_name=session_name,
            peer_name=message.peer_name,
            content=message.content,
            h_metadata=message.metadata or {},
            workspace_name=workspace_name,
            public_id=generate_nanoid(),
            token_count=len(message.encoded_message),
            created_at=message.created_at,  # Use provided created_at if available
            seq_in_session=message_seq_in_session,
        )
        message_objects.append(message_obj)

    db.add_all(message_objects)

    # Commit here to release the advisory lock before generating embeddings
    await db.commit()
    try:
        if settings.EMBED_MESSAGES:
            encoded_message_lookup = {
                msg.public_id: orig_msg.encoded_message
                for msg, orig_msg in zip(message_objects, messages, strict=True)
            }
            id_resource_dict = {
                message.public_id: (
                    message.content,
                    encoded_message_lookup[message.public_id],
                )
                for message in message_objects
            }
            embedding_dict = await embedding_client.batch_embed(id_resource_dict)

            # Get vector store and namespace for this workspace's messages
            vector_store = get_vector_store()
            namespace = vector_store.get_vector_namespace("message", workspace_name)

            # Create MessageEmbedding entries
            embedding_objects: list[models.MessageEmbedding] = []

            # Check if pgvector is being used (primary or secondary)
            # If so, write embeddings to ORM since pgvector relies on postgres
            # Otherwise, store in memory for vector store upsert only
            pgvector_in_use = (
                settings.VECTOR_STORE.PRIMARY_TYPE == "pgvector"
                or settings.VECTOR_STORE.SECONDARY_TYPE == "pgvector"
            )

            for message_obj in message_objects:
                embeddings = embedding_dict.get(message_obj.public_id, [])
                for embedding in embeddings:
                    # Create MessageEmbedding record
                    embedding_obj = models.MessageEmbedding(
                        content=message_obj.content,
                        message_id=message_obj.public_id,
                        workspace_name=workspace_name,
                        session_name=session_name,
                        peer_name=message_obj.peer_name,
                        sync_state="pending",
                    )
                    if pgvector_in_use:
                        # pgvector in use: write embedding to ORM (postgres)
                        embedding_obj.embedding = embedding
                    else:
                        # store in memory for vector store upsert only
                        embedding_obj._pending_embedding = embedding
                    embedding_objects.append(embedding_obj)

            # Add all embedding metadata objects to the session
            if embedding_objects:
                db.add_all(embedding_objects)
                await db.flush()

                # Track embedding IDs for sync state updates
                embedding_ids = [emb.id for emb in embedding_objects]

                # Build vector records - source depends on whether pgvector is in use
                vector_records: list[VectorRecord] = []
                for emb in embedding_objects:
                    if pgvector_in_use:
                        # pgvector in use: embedding is on ORM object (numpy array)
                        if emb.embedding is not None:
                            vector_records.append(
                                VectorRecord(
                                    id=str(emb.id),
                                    embedding=[float(x) for x in emb.embedding],
                                    metadata={
                                        "message_id": emb.message_id,
                                        "session_name": emb.session_name,
                                        "peer_name": emb.peer_name,
                                    },
                                )
                            )
                    else:
                        # pgvector not in use: embedding is in _pending_embedding
                        if (
                            hasattr(emb, "_pending_embedding")
                            and emb._pending_embedding is not None
                        ):
                            vector_records.append(
                                VectorRecord(
                                    id=str(emb.id),
                                    embedding=list(emb._pending_embedding),
                                    metadata={
                                        "message_id": emb.message_id,
                                        "session_name": emb.session_name,
                                        "peer_name": emb.peer_name,
                                    },
                                )
                            )
                await db.commit()

                # Retry vector upsert with exponential backoff
                if vector_records:
                    try:
                        result = None
                        async for attempt in AsyncRetrying(
                            stop=stop_after_attempt(3),
                            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
                            retry=retry_if_exception_type(Exception)
                            | retry_if_result(
                                lambda res: res is not None
                                and res.secondary_ok is False
                            ),
                            reraise=True,
                        ):
                            with attempt:
                                result = await vector_store.upsert_many(
                                    namespace, vector_records
                                )

                        if result is not None and result.secondary_ok is False:
                            # Partial success: primary has data but secondary doesn't
                            # Keep as "pending" for reconciliation to sync secondary
                            logger.warning(
                                "Partial sync for message embeddings: %s",
                                result.secondary_error,
                            )
                            await db.execute(
                                update(models.MessageEmbedding)
                                .where(models.MessageEmbedding.id.in_(embedding_ids))
                                .values(
                                    sync_attempts=models.MessageEmbedding.sync_attempts
                                    + 1,
                                    last_sync_at=func.now(),
                                )
                            )
                            await db.commit()
                        else:
                            # Success: primary succeeded and (secondary succeeded or no secondary configured)
                            await db.execute(
                                update(models.MessageEmbedding)
                                .where(models.MessageEmbedding.id.in_(embedding_ids))
                                .values(
                                    sync_state="synced",
                                    last_sync_at=func.now(),
                                    sync_attempts=0,
                                )
                            )
                            await db.commit()

                    except Exception as e:
                        # Total failure: primary write failed
                        logger.error(
                            f"Failed to upsert message vectors after 3 retries: {e}"
                        )
                        await db.execute(
                            update(models.MessageEmbedding)
                            .where(models.MessageEmbedding.id.in_(embedding_ids))
                            .values(
                                sync_attempts=models.MessageEmbedding.sync_attempts + 1,
                                last_sync_at=func.now(),
                            )
                        )
                        await db.commit()

    except Exception:
        logger.exception(
            "Failed to generate message embeddings for %s messages in workspace %s and session %s.",
            len(message_objects),
            workspace_name,
            session_name,
        )

    return message_objects


async def get_messages(
    workspace_name: str,
    session_name: str,
    reverse: bool | None = False,
    filters: dict[str, Any] | None = None,
    token_limit: int | None = None,
    message_count_limit: int | None = None,
) -> Select[tuple[models.Message]]:
    """
    Get messages from a session. If token_limit is provided, the n most recent messages
    with token count adding up to the limit will be returned. If message_count_limit is provided,
    the n most recent messages will be returned. If both are provided, message_count_limit will be
    used.

    Args:
        workspace_name: Name of the workspace
        session_name: Name of the session
        reverse: Whether to reverse the order of messages
        filters: Filter to apply to the messages
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

    # Apply message count limit first (takes precedence over token limit)
    if message_count_limit is not None:
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        # For message count limit, we want the most recent N messages
        # So we order by id desc to get most recent, then apply limit
        stmt = stmt.order_by(models.Message.id.desc()).limit(message_count_limit)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    elif token_limit is not None:
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
        stmt = apply_filter(stmt, models.Message, filters)

        # Apply final ordering based on reverse parameter
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())
    else:
        # Default case - no limits applied
        stmt = select(models.Message).where(*base_conditions)
        stmt = apply_filter(stmt, models.Message, filters)
        if reverse:
            stmt = stmt.order_by(models.Message.id.desc())
        else:
            stmt = stmt.order_by(models.Message.id.asc())

    return stmt


async def get_messages_id_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_id: int = 0,
    end_id: int | None = None,
    token_limit: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by primary key ID range.
    If end_id is not provided, all messages after and including start_id will be returned.
    If start_id is not provided, start will be beginning of session.

    Note: list is *inclusive* of the end_id message and start_id message.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_id: Primary key ID of the first message to return
        end_id: Primary key ID of the last message (exclusive)

    Returns:
        List of messages
    """
    if start_id < 0 or (end_id is not None and (start_id >= end_id or end_id <= 0)):
        return []

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]
    if end_id:
        base_conditions.append(
            and_(models.Message.id >= start_id, models.Message.id < end_id)
        )
    else:
        base_conditions.append(models.Message.id >= start_id)

    if token_limit:
        # Apply token limit logic using helper function
        stmt = _apply_token_limit(base_conditions, token_limit)
        stmt = stmt.order_by(models.Message.id)
    else:
        stmt = select(models.Message).where(*base_conditions)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_messages_by_seq_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_seq: int = 1,
    end_seq: int | None = None,
) -> list[models.Message]:
    """
    Get messages from a session by seq_in_session range.

    This is useful for getting the last N messages in a session.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        session_name: Name of the session
        start_seq: Sequence number of the first message to return (inclusive)
        end_seq: Sequence number of the last message to return (inclusive)

    Returns:
        List of messages ordered by seq_in_session
    """
    if start_seq < 1 or (end_seq is not None and start_seq > end_seq):
        return []

    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
    ]

    if end_seq is not None:
        base_conditions.append(
            and_(
                models.Message.seq_in_session >= start_seq,
                models.Message.seq_in_session <= end_seq,
            )
        )
    else:
        base_conditions.append(models.Message.seq_in_session >= start_seq)

    stmt = (
        select(models.Message)
        .where(*base_conditions)
        .order_by(models.Message.seq_in_session.asc())
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_message_seq_in_session(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: int,
) -> int:
    """
    Get the sequence number of a message within a session.

    Args:
        db: Database session
        session_name: Name of the session
        message_id: Primary key ID of the message

    Returns:
        The sequence number of the message (1-indexed)
    """
    stmt = (
        select(models.Message.seq_in_session)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name == session_name)
        .where(models.Message.id == message_id)
    )
    seq: int | None = await db.scalar(stmt)
    return int(seq) if seq is not None else 0


async def get_message(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    message_id: str,
) -> models.Message | None:
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
