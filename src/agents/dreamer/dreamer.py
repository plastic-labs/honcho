import logging

import sentry_sdk

from src.config import settings
from src.dependencies import tracked_db
from src.agents.dreamer.orchestrator import run_dream
from src.agents.dreamer.reasoning import process_reasoning_dream
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
            case DreamType.OMNI:
                async with tracked_db("dream_orchestrator") as db:
                    await run_dream(
                        db=db,
                        workspace_name=workspace_name,
                        observer=payload.observer,
                        observed=payload.observed,
                        session_name=payload.session_name,
                    )

            case DreamType.REASONING:
                async with tracked_db("reasoning_dream") as db:
                    metrics = await process_reasoning_dream(
                        db=db,
                        workspace_name=workspace_name,
                        observer=payload.observer,
                        observed=payload.observed,
                        min_observations_threshold=settings.DREAM.REASONING_MIN_OBSERVATIONS,
                        min_unfalsified_threshold=settings.DREAM.REASONING_MIN_UNFALSIFIED,
                        max_iterations=settings.DREAM.REASONING_MAX_ITERATIONS,
                    )
                    logger.info(f"Reasoning dream completed: {metrics.to_dict()}")

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.observer}/{payload.observed}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails
