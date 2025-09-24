import logging

import sentry_sdk

from src.config import settings
from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


@sentry_sdk.trace
async def process_dream(
    payload: DreamPayload,
) -> None:
    """
    Process a dream task by performing collection maintenance operations.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"Processing dream task: {payload.dream_type} for {payload.workspace_name}/{payload.target_name}"
    )

    try:
        if payload.dream_type == "consolidate":
            await _process_consolidate_dream(payload)
        ## TODO other dream types

        logger.info(
            f"Completed dream task: {payload.dream_type} for {payload.target_name}"
        )

    except Exception as e:
        logger.error(
            f"Error processing dream task {payload.dream_type} for {payload.target_name}: {str(e)}",
            exc_info=True,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        # Don't re-raise - we want to mark the dream task as processed even if it fails


async def _process_consolidate_dream(payload: DreamPayload) -> None:
    """
    Process a consolidation dream task.

    Consolidation means taking all the documents in a collection and merging
    similar observations into a single, best-quality observation document.

    TODO: need to determine a way to do this on a subset of documents since
    collections will grow very large.
    """
    logger.info(f"STUB: Consolidating documents for {payload.target_name}")

    # TODO: Implement actual consolidation logic
