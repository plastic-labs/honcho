import logging
from typing import Any

import sentry_sdk
from langfuse.decorators import langfuse_context
from pydantic import ValidationError
from rich.console import Console

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.deriver import deriver
from src.utils import summarizer
from src.utils.logging import log_performance_metrics
from src.webhooks import webhook_delivery

from .queue_payload import (
    RepresentationPayload,
    SummaryPayload,
    WebhookPayload,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)


async def process_item(task_type: str, payload: dict[str, Any]) -> None:
    """Validate an incoming queue payload and dispatch it to the appropriate handler.

    This function centralizes payload validation using a simple mapping from
    task type to Pydantic model. After validation, it routes the request to
    the correct processor without repeating type checks elsewhere.
    """
    logger.debug("process_item received payload for task type %s", task_type)

    if task_type == "webhook":
        try:
            validated = WebhookPayload(**payload)
        except ValidationError as e:
            logger.error(
                "Invalid webhook payload received: %s. Payload: %s", str(e), payload
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_webhook(validated)
        logger.debug("Finished processing webhook %s", validated.event_type)
    elif task_type == "summary":
        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_trace(  # type: ignore
                metadata={
                    "critical_analysis_model": settings.DERIVER.MODEL,
                }
            )
        try:
            validated = SummaryPayload(**payload)
        except ValidationError as e:
            logger.error(
                "Invalid summary payload received: %s. Payload: %s", str(e), payload
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_summary_task(validated)
    elif task_type == "representation":
        if settings.LANGFUSE_PUBLIC_KEY:
            langfuse_context.update_current_trace(
                metadata={
                    "critical_analysis_model": settings.DERIVER.MODEL,
                }
            )

        try:
            validated = RepresentationPayload(**payload)
        except ValidationError as e:
            logger.error(
                "Invalid representation payload received: %s. Payload: %s",
                str(e),
                payload,
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await deriver.process_representation_task(validated)
    else:
        raise ValueError(f"Invalid task type: {task_type}")


@sentry_sdk.trace
async def process_webhook(
    payload: WebhookPayload,
) -> None:
    async with tracked_db() as db:
        await webhook_delivery.deliver_webhook(db, payload)


@sentry_sdk.trace
async def process_summary_task(
    payload: SummaryPayload,
) -> None:
    """
    Process a summary task by generating summaries if needed.
    """
    message_public_id = payload.message_public_id
    # Fetch message public ID if not in payload --> fixes backward compatibility
    if not message_public_id:
        logger.debug("Fetching message public ID for message %s", payload.message_id)
        async with tracked_db("summary_fallback") as db:
            message = await crud.get_message(
                db,
                payload.workspace_name,
                payload.session_name,
                message_id=message_public_id,
            )
            if message is None:
                logger.error(
                    "Failed to fetch message with ID %s for process_summary_task",
                    payload.message_id,
                )
                return
            message_public_id = message.public_id
    await summarizer.summarize_if_needed(
        payload.workspace_name,
        payload.session_name,
        payload.message_id,
        payload.message_seq_in_session,
        payload.message_public_id,
    )
    log_performance_metrics(f"summary_{payload.workspace_name}_{payload.message_id}")
