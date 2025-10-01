import logging
from typing import Any

import sentry_sdk
from langfuse import get_client
from pydantic import ValidationError
from rich.console import Console

from src.config import settings
from src.dependencies import tracked_db
from src.deriver.deriver import process_representation_tasks_batch
from src.models import Message
from src.utils import summarizer
from src.utils.logging import log_performance_metrics
from src.webhooks import webhook_delivery

from .queue_payload import (
    SummaryPayload,
    WebhookPayload,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

lf = get_client()


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
        with sentry_sdk.start_transaction(name="process_summary_task", op="deriver"):
            if settings.LANGFUSE_PUBLIC_KEY:
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
                    )
                    log_performance_metrics(
                        f"summary_{validated.workspace_name}_{validated.message_id}"
                    )
            else:
                await summarizer.summarize_if_needed(
                    validated.workspace_name,
                    validated.session_name,
                    validated.message_id,
                    validated.message_seq_in_session,
                )
                log_performance_metrics(
                    f"summary_{validated.workspace_name}_{validated.message_id}"
                )
    else:
        raise ValueError(f"Invalid task type: {task_type}")


async def process_representation_batch(
    messages: list[Message],
    sender_name: str | None,
    target_name: str | None,
) -> None:
    """Validate incoming queue payloads and dispatch to the appropriate handler.

    This function centralizes payload validation using a simple mapping from
    task type to Pydantic model. After validation, routes the request to
    the correct processor without repeating type checks elsewhere.

    Args:
        task_type: The type of task to process
        queue_payloads: List of payload dictionaries to process
        sender_name (optional): For representation tasks, the sender_name from work_unit_key
                     to identify which messages should be focused on
        target_name (optional): For representation tasks, the target_name from work_unit_key
                     to identify which messages should be focused on
    """
    if not messages or not messages[0]:
        logger.debug("process_representation_batch received no payloads")
        return

    if sender_name is None or target_name is None:
        raise ValueError(
            "sender_name and target_name are required for representation tasks"
        )

    logger.debug(
        "process_representation_batch received %s payloads",
        len(messages),
    )

    if settings.LANGFUSE_PUBLIC_KEY:
        with lf.start_as_current_span(
            name="representation_processing",
            input={
                "payloads": [
                    {
                        "message_id": msg.id,
                        "sender_name": sender_name,
                        "target_name": target_name,
                        "session_name": msg.session_name,
                    }
                    for msg in messages
                ]
            },
            metadata={
                "critical_analysis_model": settings.DERIVER.MODEL,
            },
        ):
            await process_representation_tasks_batch(sender_name, target_name, messages)
    else:
        await process_representation_tasks_batch(sender_name, target_name, messages)
