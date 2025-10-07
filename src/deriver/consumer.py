import logging
from typing import Any

import sentry_sdk
from pydantic import ValidationError
from rich.console import Console
from sqlalchemy import select

from src import models
from src.config import settings
from src.dependencies import tracked_db
from src.deriver.deriver import process_representation_tasks_batch
from src.dreamer.dreamer import process_dream
from src.models import Message
from src.utils import summarizer
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import log_performance_metrics
from src.utils.queue_payload import (
    DreamPayload,
    SummaryPayload,
    WebhookPayload,
)
from src.webhooks import webhook_delivery

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

lf = get_langfuse_client() if settings.LANGFUSE_PUBLIC_KEY else None


async def process_item(task_type: str, queue_payload: dict[str, Any]) -> None:
    """Process a single item from the queue."""
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
            await webhook_delivery.deliver_webhook(db, validated)

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

        message_public_id = validated.message_public_id
        if not message_public_id:
            logger.info(
                "Fetching message public ID for message %s", validated.message_id
            )
            async with tracked_db(operation_name="summary_fallback") as db:
                stmt = (
                    select(models.Message)
                    .where(models.Message.workspace_name == validated.workspace_name)
                    .where(models.Message.session_name == validated.session_name)
                    .where(models.Message.id == validated.message_id)
                )
                result = await db.execute(stmt)

                message = result.scalar_one_or_none()
                if message is None:
                    logger.error(
                        "Failed to fetch message with ID %s for process_summary_task",
                        validated.message_id,
                    )
                    return
                message_public_id = message.public_id

        with sentry_sdk.start_transaction(name="process_summary_task", op="deriver"):
            if lf:
                with lf.start_as_current_span(
                    name="summary_processing",
                    input={
                        "workspace_name": validated.workspace_name,
                        "session_name": validated.session_name,
                        "message_id": validated.message_id,
                    },
                    metadata={
                        "summary_model": settings.SUMMARY.MODEL,
                    },
                ):
                    await summarizer.summarize_if_needed(
                        validated.workspace_name,
                        validated.session_name,
                        validated.message_id,
                        validated.message_seq_in_session,
                        message_public_id,
                    )
                    log_performance_metrics(
                        "summary", f"{validated.workspace_name}_{validated.message_id}"
                    )
            else:
                await summarizer.summarize_if_needed(
                    validated.workspace_name,
                    validated.session_name,
                    validated.message_id,
                    validated.message_seq_in_session,
                    message_public_id,
                )
                log_performance_metrics(
                    "summary", f"{validated.workspace_name}_{validated.message_id}"
                )

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
            await process_dream(validated)
    else:
        raise ValueError(f"Invalid task type: {task_type}")


async def process_representation_batch(
    messages: list[Message],
    *,
    observer: str | None,
    observed: str | None,
) -> None:
    """Validate incoming queue payloads and dispatch to the appropriate handler.

    This function centralizes payload validation using a simple mapping from
    task type to Pydantic model. After validation, routes the request to
    the correct processor without repeating type checks elsewhere.

    Args:
        task_type: The type of task to process
        queue_payloads: List of payload dictionaries to process
        observed (optional): For representation tasks, the observed from work_unit_key
                     to identify which messages should be focused on
        observer (optional): For representation tasks, the observer from work_unit_key
                     to identify which messages should be focused on
    """
    if not messages or not messages[0]:
        logger.debug("process_representation_batch received no payloads")
        return

    if observed is None or observer is None:
        raise ValueError("observed and observer are required for representation tasks")

    logger.debug(
        "process_representation_batch received %s payloads",
        len(messages),
    )

    if lf:
        with lf.start_as_current_span(
            name="representation_processing",
            input={
                "payloads": [
                    {
                        "message_id": msg.id,
                        "observer": observer,
                        "observed": observed,
                        "session_name": msg.session_name,
                    }
                    for msg in messages
                ]
            },
            metadata={
                "critical_analysis_model": settings.DERIVER.MODEL,
            },
        ):
            await process_representation_tasks_batch(
                messages, observer=observer, observed=observed
            )
    else:
        await process_representation_tasks_batch(
            messages, observer=observer, observed=observed
        )
