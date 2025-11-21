import logging

import sentry_sdk

from src.config import settings
from src.dreamer.agent import process_agent_dream
from src.dreamer.consolidate import process_consolidate_dream
from src.schemas import DreamType
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
    workspace_name: str,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"""
(っ- ‸ - ς)ᶻ z 𐰁 ᶻ z 𐰁 ᶻ z 𐰁\n
DREAM: {payload.dream_type} documents for {workspace_name}/{payload.observer}/{payload.observed}\n
𐰁 z ᶻ 𐰁 z ᶻ 𐰁 z ᶻ(っ- ‸ - ς)"""
    )

    try:
        match payload.dream_type:
            case DreamType.CONSOLIDATE:
                await process_consolidate_dream(payload, workspace_name)
            case DreamType.AGENT:
                await process_agent_dream(payload, workspace_name)

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails
