import logging
from typing import Any

import httpx
from pydantic import ValidationError
from rich.console import Console

from .deriver import Deriver
from .queue_payload import RepresentationPayload, SummaryPayload, WebhookQueuePayload

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

deriver = Deriver()


async def process_item(client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
    # Validate payload structure and types before processing
    try:
        task_type = payload.get("task_type")
        if task_type == "representation":
            validated_payload = RepresentationPayload(**payload)
        elif task_type == "summary":
            validated_payload = SummaryPayload(**payload)
        elif task_type == "webhook":
            validated_payload = WebhookQueuePayload(**payload)
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
    except ValidationError as e:
        logger.error("Invalid payload received: %s. Payload: %s", str(e), payload)
        raise ValueError(f"Invalid payload structure: {str(e)}") from e

    logger.debug(
        "process_item received payload for task type %s ",
        validated_payload.task_type,
    )

    if validated_payload.task_type == "webhook":
        # Type narrowing: at this point we know it's WebhookQueuePayload
        webhook_payload = validated_payload
        assert isinstance(webhook_payload, WebhookQueuePayload)
        try:
            await deriver.process_webhook(client, webhook_payload)
            logger.debug("Finished processing webhook %s", webhook_payload.event_type)
        except Exception as e:
            logger.error(
                "Failed to process webhook %s for workspace %s: %s",
                webhook_payload.event_type,
                webhook_payload.workspace_name,
                str(e),
                exc_info=True,
            )
            # Re-raise to ensure the message gets marked as processed but allows queue manager to handle the error
            raise
    else:
        # Type narrowing: at this point we know it's DeriverQueuePayload
        deriver_payload = validated_payload
        assert isinstance(deriver_payload, (RepresentationPayload, SummaryPayload))
        await deriver.process_message(deriver_payload)
        logger.debug("Finished processing message %s", deriver_payload.message_id)
