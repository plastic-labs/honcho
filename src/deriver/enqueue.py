import logging
from typing import Any

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import ValidationException
from src.models import QueueItem

from .queue_payload import DeriverQueuePayload

logger = logging.getLogger(__name__)


async def enqueue(payload: list[dict[str, Any]]) -> None:
    """
    Add message(s) to the deriver queue for processing.

    Args:
        payload: List of message payload dictionaries
    """

    # Use the get_db dependency to ensure proper transaction handling
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
                logger.info(
                    "Successfully enqueued %d messages with %d total queue items",
                    len(payload),
                    len(queue_records),
                )

        except Exception as e:
            logger.exception("Failed to enqueue messages!")
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

    deriver_disabled = bool(session.configuration.get("deriver_disabled"))

    peers_with_configuration = await get_peers_with_configuration(
        db_session, workspace_name, session_name
    )

    queue_records: list[dict[str, Any]] = []

    for message in payload:
        queue_records.extend(
            process_message(
                message,
                peers_with_configuration,
                session.id,
                deriver_disabled=deriver_disabled,
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
        row.peer_name: [row.peer_configuration, row.session_peer_configuration]
        for row in peers_with_configuration_list
    }


def create_representation_record(
    message: dict[str, Any],
    sender_name: str,
    target_name: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a queue record for representation task.

    Args:
        message: The message payload
        sender_name: Name of the sender
        target_name: Name of the target
        session_id: Optional session ID

    Returns:
        Queue record dictionary
    """
    processed_payload = DeriverQueuePayload.create_payload(
        message=message,
        sender_name=sender_name,
        target_name=target_name,
        task_type="representation",
    )
    return {
        "payload": processed_payload,
        "session_id": session_id,
    }


def create_summary_record(
    message: dict[str, Any], sender_name: str, target_name: str, session_id: str
) -> dict[str, Any]:
    """
    Create a queue record for summary task.

    Args:
        message: The message payload
        sender_name: Name of the sender
        target_name: Name of the target
        session_id: Session ID

    Returns:
        Queue record dictionary
    """
    processed_payload = DeriverQueuePayload.create_payload(
        message=message,
        sender_name=sender_name,
        target_name=target_name,
        task_type="summary",
    )
    return {
        "payload": processed_payload,
        "session_id": session_id,
    }


def get_effective_observe_me(
    sender_name: str, peers_with_configuration: dict[str, list[dict[str, Any]]]
) -> bool:
    """
    Determine the effective observe_me setting for a sender, considering session and peer configurations.

    Args:
        sender_name: Name of the sender
        peers_with_configuration: Dictionary of peer configurations

    Returns:
        True if observe_me is enabled, False otherwise
    """
    configuration = peers_with_configuration[sender_name]
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


def process_message(
    message: dict[str, Any],
    peers_with_configuration: dict[str, list[dict[str, Any]]],
    session_id: str,
    *,
    deriver_disabled: bool,
) -> list[dict[str, Any]]:
    """
    Process a single message and generate queue records based on configurations.

    Args:
        message: The message payload
        deriver_disabled: Whether deriver is disabled for the session
        peers_with_configuration: Dictionary of peer configurations
        session_id: Session ID

    Returns:
        List of queue records for this message
    """
    sender_name = message["peer_name"]

    if deriver_disabled:
        return [
            create_summary_record(
                message,
                sender_name=sender_name,
                target_name=sender_name,
                session_id=session_id,
            )
        ]

    if not get_effective_observe_me(sender_name, peers_with_configuration):
        return []

    records: list[dict[str, Any]] = [
        create_representation_record(
            message,
            sender_name=sender_name,
            target_name=sender_name,
            session_id=session_id,
        )
    ]

    for peer_name, configuration in peers_with_configuration.items():
        if peer_name == sender_name:
            continue

        session_peer_config = (
            schemas.SessionPeerConfig(**configuration[1]) if configuration[1] else None
        )

        if session_peer_config is None or not session_peer_config.observe_others:
            continue

        records.append(
            create_representation_record(
                message,
                sender_name=sender_name,
                target_name=peer_name,
                session_id=session_id,
            )
        )
        logger.debug(
            "enqueued representation task for %s's representation of %s",
            peer_name,
            sender_name,
        )

    return records
