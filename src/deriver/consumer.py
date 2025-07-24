import logging
from typing import Any

from pydantic import ValidationError
from rich.console import Console

from .deriver import Deriver
from .queue_payload import RepresentationPayload, SummaryPayload

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=True)

deriver = Deriver()


async def process_item(payload: dict[str, Any]) -> None:
    # Validate payload structure and types before processing
    try:
        task_type = payload.get("task_type")
        if task_type == "representation":
            validated_payload = RepresentationPayload(**payload)
        elif task_type == "summary":
            validated_payload = SummaryPayload(**payload)
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
    except ValidationError as e:
        logger.error("Invalid payload received: %s. Payload: %s", str(e), payload)
        raise ValueError(f"Invalid payload structure: {str(e)}") from e

    logger.debug(
        "process_item received payload for message %s in session %s, task type %s",
        validated_payload.message_id,
        validated_payload.session_name,
        validated_payload.task_type,
    )
    await deriver.process_message(validated_payload)
    logger.debug("Finished processing message %s", validated_payload.message_id)
