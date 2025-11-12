import logging
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import get_affected_dream_keys, get_dream_scheduler
from src.exceptions import ValidationException
from src.models import QueueItem
from src.utils.config_helpers import get_configuration
from src.utils.queue_payload import create_payload
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
    session_level_configuration = get_configuration(session, workspace)

    peers_with_configuration = await get_peers_with_configuration(
        db_session, workspace_name, session_name
    )

    queue_records: list[dict[str, Any]] = []

    for message in payload:
        queue_records.extend(
            await generate_queue_records(
                db_session,
                message,
                peers_with_configuration,
                session.id,
                deriver_enabled=session_level_configuration.deriver_enabled,
                summaries_enabled=session_level_configuration.summaries_enabled,
                messages_per_short_summary=session_level_configuration.messages_per_short_summary,
                messages_per_long_summary=session_level_configuration.messages_per_long_summary,
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
    session_id: str | None = None,
    *,
    observer: str,
    observed: str,
) -> dict[str, Any]:
    """
    Create a queue record for representation task.

    Args:
        message: The message payload
        observed: Name of the sender
        observer: Name of the target
        session_id: Optional session ID

    Returns:
        Queue record dictionary with workspace_name and message_id as separate fields
    """
    workspace_name = message.get("workspace_name")
    message_id = message.get("message_id")

    if not isinstance(workspace_name, str):
        raise TypeError("workspace_name is required and must be a string")
    if not isinstance(message_id, int):
        raise TypeError("message_id is required and must be an integer")

    processed_payload = create_payload(
        message=message,
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
    return sender_peer_config.observe_me


async def generate_queue_records(
    db_session: AsyncSession,
    message: dict[str, Any],
    peers_with_configuration: dict[str, list[dict[str, Any]]],
    session_id: str,
    *,
    deriver_enabled: bool,
    summaries_enabled: bool,
    messages_per_short_summary: int,
    messages_per_long_summary: int,
) -> list[dict[str, Any]]:
    """
    Process a single message and generate queue records based on configurations.

    Args:
        db_session: The database session
        message: The message payload
        deriver_enabled: Whether deriver is enabled for the session
        peers_with_configuration: Dictionary of peer configurations
        session_id: Session ID
        messages_per_short_summary: Number of messages per short summary
        messages_per_long_summary: Number of messages per long summary

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

    if summaries_enabled and (
        message_seq_in_session % messages_per_short_summary == 0
        or message_seq_in_session % messages_per_long_summary == 0
    ):
        records.append(
            create_summary_record(
                message,
                session_id=session_id,
                message_seq_in_session=message_seq_in_session,
            )
        )

    if deriver_enabled is False:
        return records

    if get_effective_observe_me(observed, peers_with_configuration):
        # global representation task
        records.append(
            create_representation_record(
                message,
                observed=observed,
                observer=observed,
                session_id=session_id,
            )
        )

        for peer_name, configuration in peers_with_configuration.items():
            if peer_name == observed:
                continue

            # If the observer peer has left the session, we don't need to enqueue a representation task for them.
            if not configuration[2]:
                continue

            session_peer_config = (
                schemas.SessionPeerConfig(**configuration[1])
                if configuration[1]
                else None
            )

            if session_peer_config is None or not session_peer_config.observe_others:
                continue

            records.append(
                # peer representation task
                create_representation_record(
                    message,
                    observed=observed,
                    observer=peer_name,
                    session_id=session_id,
                )
            )
            logger.debug(
                "enqueued representation task for %s's representation of %s",
                peer_name,
                observed,
            )

    logger.debug(
        "message %s from %s created %s queue items",
        message_id,
        observed,
        len(records),
    )

    return records
