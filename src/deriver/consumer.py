import logging
import time

import sentry_sdk
from pydantic import ValidationError
from sqlalchemy import select

from src import crud, models
from src.dependencies import tracked_db
from src.deriver.deriver import process_representation_tasks_batch
from src.dreamer import process_dream
from src.exceptions import ResourceNotFoundException
from src.models import Message
from src.reconciler.queue_cleanup import cleanup_queue_items
from src.reconciler.sync_vectors import run_vector_reconciliation_cycle
from src.schemas import ReconcilerType, ResolvedConfiguration
from src.telemetry.events import (
    CleanupStaleItemsCompletedEvent,
    DeletionCompletedEvent,
    SyncVectorsCompletedEvent,
    emit,
)
from src.telemetry.logging import log_performance_metrics
from src.utils import summarizer
from src.utils.queue_payload import (
    DeletionPayload,
    DreamPayload,
    ReconcilerPayload,
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

    # Handle reconciler first - it's the only task type that doesn't require workspace_name
    if task_type == "reconciler":
        with sentry_sdk.start_transaction(name="process_reconciler_task", op="deriver"):
            try:
                validated = ReconcilerPayload(**queue_payload)
            except ValidationError as e:
                logger.error(
                    "Invalid reconciler payload received: %s. Payload: %s",
                    str(e),
                    queue_payload,
                )
                raise ValueError(f"Invalid payload structure: {str(e)}") from e
            await process_reconciler(validated)
        return

    # All other task types require a workspace_name
    if workspace_name is None:
        raise ValueError(f"{task_type} tasks require a workspace_name")

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
    observers: list[str] | None,
    observed: str | None,
    queue_item_message_ids: list[int],
) -> None:
    """
    Prepares and processes a batch of messages for representation tasks.

    Args:
        messages: List of messages to process
        message_level_configuration: Resolved configuration for this batch
        observers: List of observers for the messages
        observed: The observed of the messages
        queue_item_message_ids: Message IDs from queue items
    """
    if not messages or not messages[0]:
        logger.debug("process_representation_batch received no messages")
        return

    if observed is None or observers is None or len(observers) == 0:
        raise ValueError("observed and observers are required for representation tasks")

    await process_representation_tasks_batch(
        messages,
        message_level_configuration,
        observers=observers,
        observed=observed,
        queue_item_message_ids=queue_item_message_ids,
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
    success = True
    error_message: str | None = None
    messages_deleted = 0
    conclusions_deleted = 0

    logger.info(
        "Processing deletion task: type=%s, resource_id=%s, workspace=%s",
        deletion_type,
        resource_id,
        workspace_name,
    )

    async with tracked_db("process_deletion") as db:
        if deletion_type == "session":
            try:
                result = await crud.delete_session(
                    db, workspace_name=workspace_name, session_name=resource_id
                )
                messages_deleted = result.messages_deleted
                conclusions_deleted = result.conclusions_deleted
                logger.info(
                    "Successfully deleted session %s in workspace %s "
                    + "(messages=%d, conclusions=%d)",
                    resource_id,
                    workspace_name,
                    messages_deleted,
                    conclusions_deleted,
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
                conclusions_deleted = 1  # Single observation deleted
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
            success = False
            error_message = f"Unsupported deletion type: {deletion_type}"
            raise ValueError(error_message)

    # Emit telemetry event
    emit(
        DeletionCompletedEvent(
            workspace_name=workspace_name,
            deletion_type=deletion_type,
            resource_id=resource_id,
            success=success,
            messages_deleted=messages_deleted,
            conclusions_deleted=conclusions_deleted,
            error_message=error_message,
        )
    )


async def process_reconciler(payload: ReconcilerPayload) -> None:
    """
    Process a reconciler task from the queue.

    Currently supports:
    - sync_vectors: Syncs pending documents/message embeddings to vector store
      and cleans up soft-deleted documents.
    - cleanup_queue: Removes old processed queue items.

    Args:
        payload: The reconciler payload containing the reconciler type
    """
    reconciler_type = payload.reconciler_type
    start_time = time.perf_counter()

    if reconciler_type == ReconcilerType.SYNC_VECTORS:
        logger.debug("Processing sync_vectors task")
        metrics = await run_vector_reconciliation_cycle()

        duration_ms = (time.perf_counter() - start_time) * 1000

        if (
            metrics.total_synced > 0
            or metrics.total_failed > 0
            or metrics.total_cleaned > 0
        ):
            logger.info(
                "Reconciliation complete: synced %s docs, %s message embeddings; failed %s docs, %s message embeddings; cleaned %s docs",
                metrics.documents_synced,
                metrics.message_embeddings_synced,
                metrics.documents_failed,
                metrics.message_embeddings_failed,
                metrics.documents_cleaned,
            )

            # Emit telemetry event
            emit(
                SyncVectorsCompletedEvent(
                    documents_synced=metrics.documents_synced,
                    documents_failed=metrics.documents_failed,
                    documents_cleaned=metrics.documents_cleaned,
                    message_embeddings_synced=metrics.message_embeddings_synced,
                    message_embeddings_failed=metrics.message_embeddings_failed,
                    total_duration_ms=duration_ms,
                )
            )

    elif reconciler_type == ReconcilerType.CLEANUP_QUEUE:
        logger.debug("Processing cleanup_queue task")
        await cleanup_queue_items()

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit telemetry event for cleanup stale items
        emit(
            CleanupStaleItemsCompletedEvent(
                total_duration_ms=duration_ms,
            )
        )

    else:
        raise ValueError(f"Unsupported reconciler type: {reconciler_type}")
