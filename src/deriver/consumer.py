import logging

import sentry_sdk
from pydantic import ValidationError
from sqlalchemy import select

from src import crud, models
from src.dependencies import tracked_db
from src.deriver.deriver import process_representation_tasks_batch
from src.dreamer.dreamer import process_dream
from src.exceptions import ResourceNotFoundException
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.utils import summarizer
from src.utils.logging import log_performance_metrics
from src.utils.queue_payload import (
    DeletionPayload,
    DreamPayload,
    SummaryPayload,
    WebhookPayload,
)
from src.webhooks import webhook_delivery

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


async def process_item(queue_item: models.QueueItem) -> None:
    """Process a single item from the queue."""
    task_type = queue_item.task_type
    queue_payload = queue_item.payload
    workspace_name = queue_item.workspace_name

    if task_type == "webhook":
        try:
            validated = WebhookPayload(**queue_payload)
        except ValidationError as e:
            logger.error(
                "Invalid webhook payload received: %s. Payload: %s",
                str(e),
                queue_payload,
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        async with tracked_db() as db:
            await webhook_delivery.deliver_webhook(db, validated, workspace_name)

    elif task_type == "summary":
        try:
            validated = SummaryPayload(**queue_payload)
        except ValidationError as e:
            logger.error(
                "Invalid summary payload received: %s. Payload: %s",
                str(e),
                queue_payload,
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e

        # Use workspace_name and message_id from QueueItem columns
        message_id = queue_item.message_id

        if message_id is None:
            raise ValueError("Summary tasks require a message_id")

        message_public_id = validated.message_public_id
        if not message_public_id:
            logger.debug("Fetching message public ID for message %s", message_id)
            async with tracked_db(operation_name="summary_fallback") as db:
                stmt = (
                    select(models.Message)
                    .where(models.Message.workspace_name == workspace_name)
                    .where(models.Message.session_name == validated.session_name)
                    .where(models.Message.id == message_id)
                )
                result = await db.execute(stmt)

                message = result.scalar_one_or_none()
                if message is None:
                    logger.error(
                        "Failed to fetch message with ID %s for process_summary_task",
                        message_id,
                    )
                    return
                message_public_id = message.public_id

        with sentry_sdk.start_transaction(name="process_summary_task", op="deriver"):
            await summarizer.summarize_if_needed(
                workspace_name,
                validated.session_name,
                message_id,
                validated.message_seq_in_session,
                message_public_id,
                validated.configuration,
            )
            log_performance_metrics("summary", f"{workspace_name}_{message_id}")

    elif task_type == "dream":
        with sentry_sdk.start_transaction(name="process_dream_task", op="deriver"):
            try:
                validated = DreamPayload(**queue_payload)
            except ValidationError as e:
                logger.error(
                    "Invalid dream payload received: %s. Payload: %s",
                    str(e),
                    queue_payload,
                )
                raise ValueError(f"Invalid payload structure: {str(e)}") from e
            await process_dream(validated, workspace_name)

    elif task_type == "deletion":
        with sentry_sdk.start_transaction(name="process_deletion_task", op="deriver"):
            try:
                validated = DeletionPayload(**queue_payload)
            except ValidationError as e:
                logger.error(
                    "Invalid deletion payload received: %s. Payload: %s",
                    str(e),
                    queue_payload,
                )
                raise ValueError(f"Invalid payload structure: {str(e)}") from e
            await process_deletion(validated, workspace_name)

    else:
        raise ValueError(f"Invalid task type: {task_type}")


async def process_representation_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observer: str | None,
    observed: str | None,
) -> None:
    """
    Prepares and processes a batch of messages for representation tasks.

    Routes to minimal or legacy deriver based on settings.DERIVER.USE_LEGACY.
    - Minimal deriver: Fast single LLM call, no peer card updates
    - Legacy deriver: Full processing with peer card updates

    Args:
        messages: List of messages to process
        message_level_configuration: Resolved configuration for this batch
        observer: The observer of the messages
        observed: The observed of the messages
    """
    if not messages or not messages[0]:
        logger.debug("process_representation_batch received no messages")
        return

    if observed is None or observer is None:
        raise ValueError("observed and observer are required for representation tasks")

    await process_representation_tasks_batch(
        messages,
        message_level_configuration,
        observer=observer,
        observed=observed,
    )


async def process_deletion(
    payload: DeletionPayload,
    workspace_name: str,
) -> None:
    """
    Process a deletion task from the queue.

    This function handles the actual deletion of resources based on the deletion type.
    It is designed to be idempotent - deleting an already-deleted resource is a no-op.

    Args:
        payload: The deletion payload containing deletion_type and resource_id
        workspace_name: The workspace name for scoping the deletion

    Raises:
        ValueError: If the deletion type is not supported
    """
    deletion_type = payload.deletion_type
    resource_id = payload.resource_id

    logger.info(
        "Processing deletion task: type=%s, resource_id=%s, workspace=%s",
        deletion_type,
        resource_id,
        workspace_name,
    )

    async with tracked_db("process_deletion") as db:
        if deletion_type == "session":
            try:
                await crud.delete_session(
                    db, workspace_name=workspace_name, session_name=resource_id
                )
                logger.info(
                    "Successfully deleted session %s in workspace %s",
                    resource_id,
                    workspace_name,
                )
            except ResourceNotFoundException as e:
                # Session not found - may have already been deleted, treat as success
                logger.warning(
                    "Session %s not found during deletion (may already be deleted): %s",
                    resource_id,
                    str(e),
                )

        elif deletion_type == "observation":
            try:
                await crud.delete_document_by_id(
                    db, workspace_name=workspace_name, document_id=resource_id
                )
                logger.info(
                    "Successfully deleted observation %s in workspace %s",
                    resource_id,
                    workspace_name,
                )
            except ResourceNotFoundException as e:
                # Document not found - may have already been deleted, treat as success
                logger.warning(
                    "Observation %s not found during deletion (may already be deleted): %s",
                    resource_id,
                    str(e),
                )

        else:
            raise ValueError(f"Unsupported deletion type: {deletion_type}")
