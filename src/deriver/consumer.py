import logging
from typing import Any

import sentry_sdk
from langfuse import get_client
from pydantic import ValidationError
from rich.console import Console

from src.config import settings
from src.dependencies import tracked_db
from src.deriver import deriver
from src.deriver.dreamer import process_dream
from src.utils import summarizer
from src.utils.logging import log_performance_metrics
from src.webhooks import webhook_delivery

from .queue_payload import (
    DreamPayload,
    RepresentationPayload,
    SummaryPayload,
    WebhookPayload,
)

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

lf = get_client()


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
            lf.update_current_trace(  # type: ignore
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
            lf.update_current_trace(
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
    elif task_type == "dream":
        try:
            validated = DreamPayload(**payload)
        except ValidationError as e:
            logger.error(
                "Invalid dream payload received: %s. Payload: %s", str(e), payload
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_dream(validated)
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
    await summarizer.summarize_if_needed(
        payload.workspace_name,
        payload.session_name,
        payload.message_id,
        payload.message_seq_in_session,
    )
    log_performance_metrics(f"summary_{payload.workspace_name}_{payload.message_id}")
