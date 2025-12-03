import logging

from src.utils.queue_payload import DreamPayload

logger = logging.getLogger(__name__)


async def process_agent_dream(payload: DreamPayload, workspace_name: str) -> None:
    """
    Process an agent dream task.

    Args:
        payload: The dream task payload containing workspace, peer, and dream type information
    """
    logger.info(
        f"Processing agent dream for {workspace_name}/{payload.observer}/{payload.observed}"
    )
