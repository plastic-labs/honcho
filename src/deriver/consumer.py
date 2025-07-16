import logging
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.summarizer import summarize_if_needed

from .deriver import Deriver
from .queue_payload import DeriverQueuePayload

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

deriver = Deriver()


async def process_item(db: AsyncSession, payload: dict[str, Any]):
    # Validate payload structure and types before processing
    try:
        validated_payload = DeriverQueuePayload(**payload)
    except ValidationError as e:
        logger.error("Invalid payload received: %s. Payload: %s", str(e), payload)
        raise ValueError(f"Invalid payload structure: {str(e)}") from e

    logger.debug(
        "process_item received payload for message %s in session %s",
        validated_payload.message_id,
        validated_payload.session_name,
    )

    if validated_payload.task_type == "representation":
        logger.debug(
            "Processing message %s in %s",
            validated_payload.message_id,
            validated_payload.session_name,
        )
        await deriver.process_message(validated_payload)
        logger.debug(
            "Finished processing message %s in %s %s",
            validated_payload.message_id,
            "session" if validated_payload.session_name else "peer",
            (
                validated_payload.session_name
                if validated_payload.session_name
                else validated_payload.sender_name
            ),
        )
    await summarize_if_needed(
        db,
        validated_payload.workspace_name,
        validated_payload.session_name,
        validated_payload.sender_name,
        validated_payload.message_id,
    )
    return
