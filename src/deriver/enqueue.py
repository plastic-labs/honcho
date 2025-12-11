import logging
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import get_affected_dream_keys, get_dream_scheduler
from src.exceptions import ValidationException
from src.models import QueueItem
from src.schemas import MessageConfiguration, ResolvedConfiguration
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import (
    ReasoningFocus,
    create_deletion_payload,
    create_dream_payload,
    create_payload,
)
from src.utils.work_unit import construct_work_unit_key

logger = logging.getLogger(__name__)


async def enqueue(payload: list[dict[str, Any]]) -> None:
    """
    Add message(s) to the deriver queue for processing.

    Args:
        payload: List of message payload dictionaries
    """

    # Cancel any pending dreams for affected collections since user is active again
    dream_scheduler = get_dream_scheduler()
    if dream_scheduler and payload:
        cancelled_dreams: set[str] = set()
        for message in payload:
            # Generate work unit keys for dreams that might be affected by this message
            dream_keys: list[str] = get_affected_dream_keys(message)
            for dream_key in dream_keys:
                if await dream_scheduler.cancel_dream(dream_key):
                    cancelled_dreams.add(dream_key)

        if cancelled_dreams:
            logger.info(
                f"Cancelled {len(cancelled_dreams)} pending dreams due to new activity"
            )

    async with tracked_db("message_enqueue") as db_session:
        try:
            # Determine if batch or single processing
            if not payload:  # Empty list check
                return
            workspace_name = payload[0]["workspace_name"]
            session_name = payload[0]["session_name"]

            if session_name is None or workspace_name is None:
                raise ValidationException("Session and workspace are required")

            queue_records = await handle_session(
                db_session, payload, workspace_name, session_name
            )

            if queue_records:
                stmt = insert(QueueItem).returning(QueueItem)
                await db_session.execute(stmt, queue_records)
                await db_session.commit()

        except Exception as e:
            logger.exception("Failed to enqueue message(s)!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)


async def handle_session(
    db_session: AsyncSession,
    payload: list[dict[str, Any]],
    workspace_name: str,
    session_name: str,
) -> list[dict[str, Any]]:
    """
    Handle enqueueing for normal session cases, creating appropriate queue items based on configurations.

    Args:
        db_session: The database session
        payload: List of message payloads
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        List of queue records to insert
    """
    session = await crud.get_or_create_session(
        db_session,
        session=schemas.SessionCreate(name=session_name),
        workspace_name=workspace_name,
    )

    # Fetch workspace for configuration resolution
    workspace = await crud.get_workspace(db_session, workspace_name=workspace_name)

    # Resolve summary configuration with hierarchical fallback
    session_level_configuration = get_configuration(None, session, workspace)

    peers_with_configuration = await get_peers_with_configuration(
        db_session, workspace_name, session_name
    )

    queue_records: list[dict[str, Any]] = []

    for message in payload:
        message_config: MessageConfiguration | None = message.get("configuration")
        if message_config is not None:
            message_level_configuration = get_configuration(
                message_config, session, workspace
            )
        else:
            message_level_configuration = session_level_configuration
        queue_records.extend(
            await generate_queue_records(
                db_session,
                message,
                peers_with_configuration,
                session.id,
                message_level_configuration,
            )
        )
    return queue_records


async def get_peers_with_configuration(
    db_session: AsyncSession, workspace_name: str, session_name: str
) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve peers with their configurations for a given session.

    Args:
        db_session: The database session
        workspace_name: Name of the workspace
        session_name: Name of the session

    Returns:
        Dictionary mapping peer names to their configurations
    """
    configuration_query = await crud.get_session_peer_configuration(
        workspace_name=workspace_name, session_name=session_name
    )
    peers_with_configuration_result = await db_session.execute(configuration_query)
    peers_with_configuration_list = peers_with_configuration_result.all()
    return {
        row.peer_name: [
            row.peer_configuration,
            row.session_peer_configuration,
            row.is_active,
        ]
        for row in peers_with_configuration_list
    }


def create_representation_record(
    message: dict[str, Any],
    conf: ResolvedConfiguration,
    session_id: str | None = None,
    *,
    observer: str,
    observed: str,
) -> dict[str, Any]:
    """
    Create a queue record for representation task.

    Args:
        message: The message payload
        conf: Resolved configuration for this particular message
        session_id: Optional session ID
        observed: Name of the sender
        observer: Name of the target

    Returns:
        Queue record dictionary with workspace_name and message_id as separate fields
    """
    workspace_name = message.get("workspace_name")
    message_id = message.get("message_id")

    if not isinstance(workspace_name, str):
        raise TypeError("workspace_name is required and must be a string")
    if not isinstance(message_id, int):
        raise TypeError("message_id is required and must be an integer")

    processed_payload: dict[str, Any] = create_payload(
        message=message,
        configuration=conf,
        task_type="representation",
        observer=observer,
        observed=observed,
    )
    return {
        "work_unit_key": construct_work_unit_key(workspace_name, processed_payload),
        "payload": processed_payload,
        "session_id": session_id,
        "task_type": "representation",
        "workspace_name": workspace_name,
        "message_id": message_id,
    }


def create_summary_record(
    message: dict[str, Any],
    configuration: ResolvedConfiguration,
    session_id: str,
    message_seq_in_session: int,
) -> dict[str, Any]:
    """
    Create a queue record for summary task.

    Args:
        message: The message payload
        session_id: Session ID
        message_seq_in_session: The sequence number of the message in the session

    Returns:
        Queue record dictionary with workspace_name and message_id as separate fields
    """
    workspace_name = message.get("workspace_name")
    message_id = message.get("message_id")

    if not isinstance(workspace_name, str):
        raise ValueError("workspace_name is required and must be a string")
    if not isinstance(message_id, int):
        raise ValueError("message_id is required and must be an integer")

    processed_payload = create_payload(
        message=message,
        configuration=configuration,
        task_type="summary",
        message_seq_in_session=message_seq_in_session,
    )
    return {
        "work_unit_key": construct_work_unit_key(workspace_name, processed_payload),
        "payload": processed_payload,
        "session_id": session_id,
        "task_type": "summary",
        "workspace_name": workspace_name,
        "message_id": message_id,
    }


def get_effective_observe_me(
    observed: str, peers_with_configuration: dict[str, list[dict[str, Any]]]
) -> bool:
    """
    Determine the effective observe_me setting for a sender, considering session and peer configurations.

    Args:
        observed: Name of the sender
        peers_with_configuration: Dictionary of peer configurations

    Returns:
        True if observe_me is enabled, False otherwise
    """
    # If the sender is not in peers_with_configuration, they left after sending a message.
    # We'll use the default behavior of observing the sender by instantiating the default
    # peer-level and session-level configs.
    configuration: list[Any] = peers_with_configuration.get(observed, [{}, {}])
    sender_session_peer_config = (
        schemas.SessionPeerConfig(**configuration[1]) if configuration[1] else None
    )
    sender_peer_config = (
        schemas.PeerConfig(**configuration[0])
        if configuration[0]
        else schemas.PeerConfig()
    )

    # Session peer config takes precedence if it exists and has observe_me set
    if sender_session_peer_config and sender_session_peer_config.observe_me is not None:
        return sender_session_peer_config.observe_me

    # Otherwise use peer config
    return (
        sender_peer_config.observe_me
        if sender_peer_config.observe_me is not None
        else True
    )


async def generate_queue_records(
    db_session: AsyncSession,
    message: dict[str, Any],
    peers_with_configuration: dict[str, list[dict[str, Any]]],
    session_id: str,
    conf: ResolvedConfiguration,
) -> list[dict[str, Any]]:
    """
    Process a single message and generate queue records based on configurations.

    Args:
        db_session: The database session
        message: The message payload
        peers_with_configuration: Dictionary of peer configurations
        session_id: Session ID
        configuration: Resolved configuration for this particular message

    Returns:
        List of queue records for this message
    """
    observed = message["peer_name"]
    message_id: int = message["message_id"]

    # Prefer the sequence captured during message creation; fallback only if missing
    message_seq_in_session = int(message.get("seq_in_session") or 0)
    if message_seq_in_session <= 0:
        message_seq_in_session = await crud.get_message_seq_in_session(
            db_session,
            workspace_name=message["workspace_name"],
            session_name=message["session_name"],
            message_id=message_id,
        )

    records: list[dict[str, Any]] = []

    if conf.summary.enabled and (
        message_seq_in_session % conf.summary.messages_per_short_summary == 0
        or message_seq_in_session % conf.summary.messages_per_long_summary == 0
    ):
        records.append(
            create_summary_record(
                message,
                configuration=conf,
                session_id=session_id,
                message_seq_in_session=message_seq_in_session,
            )
        )

    # Check if the sender should be observed based on peer configuration
    should_observe = get_effective_observe_me(observed, peers_with_configuration)

    if not conf.deriver.enabled:
        return records

    if should_observe:
        # global representation task
        records.append(
            create_representation_record(
                message,
                conf,
                observed=observed,
                observer=observed,
                session_id=session_id,
            )
        )

        for peer_name, peer_conf in peers_with_configuration.items():
            if peer_name == observed:
                continue

            # If the observer peer has left the session, we don't need to enqueue a representation task for them.
            if not peer_conf[2]:
                continue

            session_peer_config = (
                schemas.SessionPeerConfig(**peer_conf[1]) if peer_conf[1] else None
            )

            if session_peer_config is None or not session_peer_config.observe_others:
                continue

            records.append(
                # peer representation task
                create_representation_record(
                    message,
                    conf,
                    observed=observed,
                    observer=peer_name,
                    session_id=session_id,
                )
            )

    logger.debug(
        "message %s from %s created %s queue items",
        message_id,
        observed,
        len(records),
    )

    return records


def create_dream_record(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    dream_type: schemas.DreamType,
    session_name: str,
    reasoning_focus: ReasoningFocus | None = None,
) -> dict[str, Any]:
    """
    Create a queue record for a dream task.

    Args:
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        dream_type: Type of dream to execute
        session_name: Name of the session to scope the dream to
        reasoning_focus: Optional focus mode for the dream ('deduction', 'induction', 'consolidation')

    Returns:
        Queue record dictionary with workspace_name and other fields
    """
    dream_payload = create_dream_payload(
        dream_type,
        observer=observer,
        observed=observed,
        session_name=session_name,
        reasoning_focus=reasoning_focus,
    )

    return {
        "work_unit_key": construct_work_unit_key(workspace_name, dream_payload),
        "payload": dream_payload,
        "session_id": None,
        "task_type": "dream",
        "workspace_name": workspace_name,
        "message_id": None,
    }


async def enqueue_dream(
    workspace_name: str,
    observer: str,
    observed: str,
    dream_type: schemas.DreamType,
    document_count: int,
    session_name: str,
    reasoning_focus: ReasoningFocus | None = None,
) -> None:
    """
    Enqueue a dream task for immediate processing by the deriver.

    Args:
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        dream_type: Type of dream to execute
        document_count: Current document count for metadata update
        session_name: Name of the session to scope the dream to
        reasoning_focus: Optional focus mode for the dream ('deduction', 'induction', 'consolidation')
    """
    async with tracked_db("dream_enqueue") as db_session:
        try:
            # Create the dream queue record
            dream_record = create_dream_record(
                workspace_name,
                observer=observer,
                observed=observed,
                dream_type=dream_type,
                session_name=session_name,
                reasoning_focus=reasoning_focus,
            )

            # Insert into queue
            stmt = insert(QueueItem).returning(QueueItem)
            await db_session.execute(stmt, [dream_record])

            # Update collection metadata
            now_iso = datetime.now(timezone.utc).isoformat()
            update_stmt = (
                update(models.Collection)
                .where(
                    models.Collection.workspace_name == workspace_name,
                    models.Collection.observer == observer,
                    models.Collection.observed == observed,
                )
                .values(
                    internal_metadata=models.Collection.internal_metadata.op("||")(
                        {
                            "dream": {
                                "last_dream_document_count": document_count,
                                "last_dream_at": now_iso,
                            }
                        }
                    )
                )
            )
            await db_session.execute(update_stmt)
            await db_session.commit()

            focus_str = f", focus: {reasoning_focus.value}" if reasoning_focus else ""
            logger.info(
                "Enqueued dream task for %s/%s/%s (type: %s%s)",
                workspace_name,
                observer,
                observed,
                dream_type.value,
                focus_str,
            )

        except Exception as e:
            logger.exception("Failed to enqueue dream task!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            raise


def create_deletion_record(
    workspace_name: str,
    deletion_type: Literal["session", "observation"],
    resource_id: str,
) -> dict[str, Any]:
    """
    Create a queue record for a deletion task.

    Args:
        workspace_name: Name of the workspace
        deletion_type: Type of resource to delete ("session" or "observation")
        resource_id: ID of the resource to delete

    Returns:
        Queue record dictionary for insertion into the queue
    """
    deletion_payload = create_deletion_payload(
        deletion_type=deletion_type,
        resource_id=resource_id,
    )

    return {
        "work_unit_key": construct_work_unit_key(workspace_name, deletion_payload),
        "payload": deletion_payload,
        "session_id": None,
        "task_type": "deletion",
        "workspace_name": workspace_name,
        "message_id": None,
    }


async def enqueue_deletion(
    workspace_name: str,
    deletion_type: Literal["session", "observation"],
    resource_id: str,
    db_session: AsyncSession | None = None,
) -> None:
    """
    Enqueue a deletion task for processing by the deriver.

    This function adds a deletion task to the queue for asynchronous processing.
    The deletion will be handled by the queue consumer with retry support.

    Args:
        workspace_name: Name of the workspace
        deletion_type: Type of resource to delete ("session" or "observation")
        resource_id: ID of the resource to delete
        db_session: Optional database session. If provided, uses this session
            instead of creating a new one. The caller is responsible for committing.
    """

    async def _do_enqueue(session: AsyncSession, should_commit: bool) -> None:
        deletion_record = create_deletion_record(
            workspace_name,
            deletion_type,
            resource_id,
        )

        stmt = insert(QueueItem).returning(QueueItem)
        await session.execute(stmt, [deletion_record])

        if should_commit:
            await session.commit()

        logger.info(
            "Enqueued deletion task: type=%s, resource_id=%s, workspace=%s",
            deletion_type,
            resource_id,
            workspace_name,
        )

    try:
        if db_session is not None:
            # Use the provided session - caller is responsible for committing
            await _do_enqueue(db_session, should_commit=False)
        else:
            # Create a new session and commit
            async with tracked_db("deletion_enqueue") as new_session:
                await _do_enqueue(new_session, should_commit=True)

    except Exception as e:
        logger.exception("Failed to enqueue deletion task!")
        if settings.SENTRY.ENABLED:
            import sentry_sdk

            sentry_sdk.capture_exception(e)
        raise
