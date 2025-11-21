import logging

from src import crud, models, schemas
from src.agent.core import Agent
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)


async def process_agent_task(queue_item: models.QueueItem) -> None:
    """
    Process a single agent task from the queue.
    This is a legacy wrapper maintained for backwards compatibility.
    Consider using process_agent_task_batch for better performance.
    """
    payload = queue_item.payload
    message_id = queue_item.message_id
    observer = payload.get("observer")
    observed = payload.get("observed")

    if not message_id or not observer or not observed:
        logger.error("Agent task requires a message_id, observer, and observed")
        return

    # Parse configuration
    config_dict = payload.get("configuration")
    if config_dict:
        configuration = schemas.ResolvedConfiguration.model_validate(config_dict)
    else:
        logger.warning("No configuration found in payload, using defaults")
        raise ValueError("Configuration missing in agent task payload")

    async with tracked_db("process_agent_task") as db:
        # Fetch the message
        message = await db.get(models.Message, message_id)

        if not message:
            logger.error(f"Message {message_id} not found")
            return

        # For single message processing, use the batch processor
        await process_agent_task_batch(
            [message], configuration, observer=observer, observed=observed
        )


async def process_agent_task_batch(
    messages: list[models.Message],
    message_level_configuration: schemas.ResolvedConfiguration | None,
    *,
    observer: str,
    observed: str,
) -> None:
    """
    Process a batch of agent tasks in a single agentic run.
    The observer/observed relationship is fixed in the queue item itself.

    Args:
        messages: List of messages to process together
        message_level_configuration: Resolved configuration for this batch
        observer: The peer who is observing (from queue item)
        observed: The peer being observed (from queue item)
    """
    if not messages or not messages[0]:
        logger.debug("process_agent_task_batch received no messages")
        return

    if message_level_configuration is None:
        logger.error("Agent tasks require configuration")
        raise ValueError("Configuration is required for agent tasks")

    logger.info(
        "Processing agent batch with %d messages (observer=%s, observed=%s)",
        len(messages),
        observer,
        observed,
    )

    # Get the workspace and session from the first message (all should be the same session)
    workspace_name = messages[0].workspace_name
    session_name = messages[0].session_name

    async with tracked_db("process_agent_task_batch") as db:
        try:
            # Fetch peer card if configuration allows
            observed_peer_card: list[str] | None = None
            if message_level_configuration.peer_card.use is not False:
                observed_peer_card = await crud.get_peer_card(
                    db,
                    workspace_name,
                    observer=observer,
                    observed=observed,
                )

            # Initialize agent with the fixed observer/observed relationship from queue item
            agent = Agent(
                db=db,
                workspace_name=workspace_name,
                session_name=session_name,
                configuration=message_level_configuration,
                observer=observer,
                observed=observed,
                observed_peer_card=observed_peer_card,
            )

            # Run agent loop for the message batch
            await agent.run_loop(messages)

        except Exception as e:
            logger.error(
                f"Error processing agent batch: {e}",
                exc_info=True,
            )
            raise
