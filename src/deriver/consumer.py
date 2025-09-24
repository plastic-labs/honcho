import logging
from typing import Any

import sentry_sdk
from pydantic import ValidationError
from rich.console import Console

from src.config import settings
from src.dependencies import tracked_db
from src.deriver.deriver import process_representation_tasks_batch
from src.dreamer.dreamer import process_dream
from src.utils import summarizer
from src.utils.langfuse_client import get_langfuse_client
from src.utils.logging import log_performance_metrics
from src.utils.queue_payload import (
    DreamPayload,
    RepresentationPayload,
    RepresentationPayloads,
    SummaryPayload,
    WebhookPayload,
)
from src.webhooks import webhook_delivery

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

lf = get_langfuse_client()


async def process_items(task_type: str, queue_payloads: list[dict[str, Any]]) -> None:
    """Validate incoming queue payloads and dispatch to the appropriate handler.

    This function centralizes payload validation using a simple mapping from
    task type to Pydantic model. After validation, routes the request to
    the correct processor without repeating type checks elsewhere.
    """
    if not queue_payloads or not queue_payloads[0]:
        logger.debug("process_items received no payloads for task type %s", task_type)
        return

    logger.debug(
        "process_items received %s payloads for task type %s",
        len(queue_payloads),
        task_type,
    )

    if task_type == "webhook":
        if len(queue_payloads) > 1:
            logger.error(
                "Received multiple webhook payloads for task type %s. Only the first one will be processed.",
                task_type,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_message(
                    "Received multiple webhook payloads for task type %s. Only the first one will be processed.",
                )
        try:
            validated = WebhookPayload(**queue_payloads[0])
        except ValidationError as e:
            logger.error(
                "Invalid webhook payload received: %s. Payload: %s",
                str(e),
                queue_payloads[0],
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_webhook(validated)
        logger.debug("Finished processing webhook %s", validated.event_type)

    elif task_type == "summary":
        if len(queue_payloads) > 1:
            logger.error(
                "Received multiple summary payloads for task type %s. Only the first one will be processed.",
                task_type,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_message(
                    "Received multiple summary payloads for task type %s. Only the first one will be processed.",
                )
        if settings.LANGFUSE_PUBLIC_KEY:
            lf.update_current_trace(  # type: ignore
                metadata={
                    "critical_analysis_model": settings.DERIVER.MODEL,
                }
            )
        try:
            validated = SummaryPayload(**queue_payloads[0])
        except ValidationError as e:
            logger.error(
                "Invalid summary payload received: %s. Payload: %s",
                str(e),
                queue_payloads[0],
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
            validated_payloads = RepresentationPayloads(
                payloads=[
                    RepresentationPayload(**payload) for payload in queue_payloads
                ]
            )
        except ValidationError as e:
            logger.error(
                "Invalid representation payloads received: %s. Payloads: %s",
                str(e),
                queue_payloads,
            )
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_representation_tasks_batch(validated_payloads.payloads)

    elif task_type == "dream":
        if len(queue_payloads) > 1:
            logger.error(
                "Received multiple dream payloads for task type %s. Only the first one will be processed.",
                task_type,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_message(
                    "Received multiple dream payloads for task type %s. Only the first one will be processed.",
                )
        try:
            validated = DreamPayload(**queue_payloads[0])
        except ValidationError as e:
            logger.error(
                "Invalid dream payload received: %s. Payload: %s",
                str(e),
                queue_payloads[0],
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
