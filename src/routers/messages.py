import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy.sql import insert

from src import crud, schemas
from src.dependencies import db, tracked_db
from src.exceptions import ResourceNotFoundException
from src.models import QueueItem
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions/{session_id}/messages",
    tags=["messages"],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)


def create_processed_payload(
    message: dict,
    sender_name: Optional[str],
    target_name: Optional[str],
    task_type: str,
) -> dict:
    """
    Create a processed payload from a message for queue processing.

    Args:
        message: The original message dictionary
        sender_name: Name of the message sender
        target_name: Name of the target peer
        task_type: Type of task ('representation' or 'summary')

    Returns:
        Processed payload dictionary ready for queue processing
    """
    processed_payload = {
        k: str(v) if isinstance(v, str) else v for k, v in message.items()
    }
    # Remove peer_name from payload
    processed_payload.pop("peer_name", None)  # Use None as default to avoid KeyError
    processed_payload["sender_name"] = sender_name
    processed_payload["target_name"] = target_name
    processed_payload["task_type"] = task_type
    return processed_payload


async def enqueue(payload: list[dict]):
    """
    Add message(s) to the deriver queue for processing.

    Args:
        payload: Single message payload or list of message payloads
    """

    # Use the get_db dependency to ensure proper transaction handling
    async with tracked_db("message_enqueue") as db_session:
        try:
            # Determine if batch or single processing
            if not payload:  # Empty list check
                logger.debug("Empty payload list, skipping enqueue")
                return
            logger.debug(f"Enqueueing batch of {len(payload)} messages")
            workspace_name = payload[0]["workspace_name"]
            session_name = payload[0]["session_name"]

            # Case 1: session_name is None — only create representation for peer
            if session_name is None:
                peer_name = payload[0]["peer_name"]
                logger.info(
                    "Session name is None, creating single representation queue items"
                )
                peer = await crud.get_or_create_peers(
                    db_session,
                    workspace_name=workspace_name,
                    peers=[schemas.PeerCreate(name=peer_name)],
                )
                peer = peer[0]

                # Cast configuration to PeerConfig and check observe_me
                peer_config = (
                    schemas.PeerConfig(**peer.configuration)
                    if peer.configuration
                    else schemas.PeerConfig()
                )
                if not peer_config.observe_me:
                    logger.info(
                        f"Peer {peer_name} has observe_me=False, skipping enqueue"
                    )
                    return

                queue_records: list[dict[str, Any]] = []

                for message in payload:
                    processed_payload = create_processed_payload(
                        message=message,
                        sender_name=message["peer_name"],
                        target_name=message["peer_name"],
                        task_type="representation",
                    )
                    queue_records.append(
                        {
                            "payload": processed_payload,
                            "session_id": None,
                        }
                    )

                logger.debug(
                    f"Inserting {len(queue_records)} queue records for None session"
                )
                stmt = insert(QueueItem).returning(QueueItem)
                await db_session.execute(stmt, queue_records)
                await db_session.commit()
                logger.info(
                    f"Successfully enqueued {len(queue_records)} messages with None session"
                )
                return

            # Case 2: Normal session processing
            session = await crud.get_or_create_session(
                db_session,
                session=schemas.SessionCreate(name=session_name),
                workspace_name=workspace_name,
            )

            # Check if deriver is disabled for this session
            deriver_disabled = (
                session.configuration.get("deriver_disabled") is not None
                and session.configuration.get("deriver_disabled") is not False
            )

            configuration_query = await crud.get_session_peer_configuration(
                workspace_name=workspace_name, session_name=session_name
            )
            peers_with_configuration_result = await db_session.execute(
                configuration_query
            )
            peers_with_configuration_list = peers_with_configuration_result.all()
            peers_with_configuration = {
                row.peer_name: [row.peer_configuration, row.session_peer_configuration]
                for row in peers_with_configuration_list
            }

            # Process all payloads - create multiple queue items per message
            queue_records = []

            for message in payload:
                if deriver_disabled:
                    # still create a summary queue item for the session
                    processed_payload = create_processed_payload(
                        message=message,
                        sender_name=None,
                        target_name=None,
                        task_type="summary",
                    )
                    queue_records.append(
                        {
                            "payload": processed_payload,
                            "session_id": session.id,
                        }
                    )
                    continue

                sender_name = message["peer_name"]

                sender_session_peer_config = (
                    schemas.SessionPeerConfig(
                        **peers_with_configuration[sender_name][1]
                    )
                    if peers_with_configuration[sender_name][1]
                    else None
                )
                sender_peer_config = (
                    schemas.PeerConfig(**peers_with_configuration[sender_name][0])
                    if peers_with_configuration[sender_name][0]
                    else schemas.PeerConfig()
                )

                observe_me = (
                    (
                        sender_session_peer_config.observe_me
                        if sender_session_peer_config.observe_me is not None
                        else sender_peer_config.observe_me
                    )
                    if sender_session_peer_config
                    else sender_peer_config.observe_me
                )
                if not observe_me:
                    continue

                # Handle working representation for sender
                processed_payload = create_processed_payload(
                    message=message,
                    sender_name=sender_name,
                    target_name=sender_name,
                    task_type="representation",
                )

                queue_records.append(
                    {
                        "payload": processed_payload,
                        "session_id": session.id,
                    }
                )
                for peer_name, configuration in peers_with_configuration.items():
                    session_peer_config = (
                        schemas.SessionPeerConfig(**configuration[1])
                        if configuration[1]
                        else None
                    )

                    if peer_name != sender_name:
                        # Handle local representation for other peers
                        if (
                            session_peer_config is None
                            or not session_peer_config.observe_others
                        ):
                            continue
                        else:
                            # Create local representation for peer
                            processed_payload = create_processed_payload(
                                message=message,
                                sender_name=sender_name,
                                target_name=peer_name,
                                task_type="representation",
                            )

                            queue_records.append(
                                {
                                    "payload": processed_payload,
                                    "session_id": session.id,
                                }
                            )

            logger.debug(f"Inserting {len(queue_records)} queue records")

            if len(queue_records) > 0:
                # Use insert to maintain order
                stmt = insert(QueueItem).returning(QueueItem)
                await db_session.execute(stmt, queue_records)
                await db_session.commit()

                logger.info(
                    f"Successfully enqueued {len(payload)} messages with {len(queue_records)} total queue items"
                )

        except Exception as e:
            logger.error(f"Failed to enqueue messages: {str(e)}", exc_info=True)
            if os.getenv("SENTRY_ENABLED", "False").lower() == "true":
                import sentry_sdk

                sentry_sdk.capture_exception(e)


@router.post("/", response_model=list[schemas.Message])
async def create_messages_for_session(
    background_tasks: BackgroundTasks,
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    messages: schemas.MessageBatchCreate = Body(
        ..., description="Batch of messages to create"
    ),
    db=db,
):
    workspace_name, session_name = workspace_id, session_id
    """Bulk create messages for a session while maintaining order. Maximum 100 messages per batch."""
    try:
        created_messages = await crud.create_messages(
            db,
            messages=messages.messages,
            workspace_name=workspace_name,
            session_name=session_name,
        )

        # Create payloads for all messages
        payloads = [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": message.id,
                "content": message.content,
                "peer_name": message.peer_name,
            }
            for message in created_messages
        ]

        # Enqueue all messages in one call
        background_tasks.add_task(enqueue, payloads)  # type: ignore
        logger.info(
            f"Batch of {len(created_messages)} messages created and queued for processing"
        )

        return created_messages
    except ValueError as e:
        logger.error(
            f"Failed to create batch messages for session {session_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Session not found") from e


@router.post("/list", response_model=Page[schemas.Message])
async def get_messages(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    options: Optional[schemas.MessageGet] = Body(
        None, description="Filtering options for the messages list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    db=db,
):
    """Get all messages for a session"""
    try:
        filter = None
        if options and hasattr(options, "filter"):
            filter = options.filter
            if filter == {}:
                filter = None

        messages_query = await crud.get_messages(
            workspace_name=workspace_id,
            session_name=session_id,
            filter=filter,
            reverse=reverse,
        )

        return await paginate(db, messages_query)
    except ValueError as e:
        logger.warning(f"Failed to get messages for session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get("/{message_id}", response_model=schemas.Message)
async def get_message(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    message_id: str = Path(..., description="ID of the message to retrieve"),
    db=db,
):
    """Get a Message by ID"""
    honcho_message = await crud.get_message(
        db, workspace_name=workspace_id, session_name=session_id, message_id=message_id
    )
    if honcho_message is None:
        logger.warning(f"Message {message_id} not found in session {session_id}")
        raise ResourceNotFoundException(f"Message with ID {message_id} not found")
    return honcho_message


@router.put("/{message_id}", response_model=schemas.Message)
async def update_message(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    message_id: str = Path(..., description="ID of the message to update"),
    message: schemas.MessageUpdate = Body(
        ..., description="Updated message parameters"
    ),
    db=db,
):
    """Update the metadata of a Message"""
    try:
        updated_message = await crud.update_message(
            db,
            message=message,
            workspace_name=workspace_id,
            session_name=session_id,
            message_id=message_id,
        )
        logger.info(f"Message {message_id} updated successfully")
        return updated_message
    except ValueError as e:
        logger.warning(f"Failed to update message {message_id}: {str(e)}")
        raise ResourceNotFoundException("Message not found") from e
