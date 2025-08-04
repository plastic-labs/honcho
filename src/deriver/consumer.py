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


async def process_item(
    client: httpx.AsyncClient, task_type: str, payload: dict[str, Any]
) -> None:
    # Validate payload structure and types before processing
    try:
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
        task_type,
    )

    if task_type == "webhook":
        if not isinstance(validated_payload, WebhookQueuePayload):
            raise ValueError(
                f"Expected WebhookQueuePayload for webhook task, got {type(validated_payload)}"
            )
        await deriver.process_webhook(client, validated_payload)
        logger.debug("Finished processing webhook %s", validated_payload.event_type)
    else:
        if not isinstance(validated_payload, RepresentationPayload | SummaryPayload):
            raise ValueError(
                f"Expected DeriverQueuePayload for task_type '{task_type}', got {type(validated_payload)}"
            )
        deriver_payload = validated_payload
        await deriver.process_message(task_type, deriver_payload)
        logger.debug("Finished processing message %s", deriver_payload.message_id)
