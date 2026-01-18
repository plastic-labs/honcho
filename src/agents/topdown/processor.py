"""Task processor for top-down reasoning agents.

⚠️  DEPRECATED: This module is no longer used for real-time task processing.
    Top-down reasoning now happens during reasoning dreams.
    See src/agents/dreamer/reasoning.py for the new dream-based workflow.

This module previously handled execution of queued tasks for Abducer, Predictor,
Falsifier, and Inductor agents via the queue consumer. Now these agents are
called directly from the reasoning dream orchestrator.

The automatic chaining logic below is preserved for reference but is not actively used.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.abducer import AbducerAgent, AbducerConfig
from src.agents.falsifier import FalsifierAgent, FalsifierConfig
from src.agents.inductor import InductorAgent, InductorConfig
from src.agents.predictor import PredictorAgent, PredictorConfig
from src.agents.topdown.enqueue import (
    enqueue_falsification,
    enqueue_induction,
    enqueue_prediction_testing,
)

logger = logging.getLogger(__name__)


async def process_hypothesis_generation(
    db_session: AsyncSession,
    payload: dict[str, Any],
    task_id: int,
) -> dict[str, Any]:
    """
    Process a hypothesis generation task using the Abducer agent.

    Args:
        db_session: Database session
        payload: Task payload containing workspace_name, observer, observed
        task_id: ID of the current task

    Returns:
        Result dictionary with hypothesis_ids and metrics
    """
    workspace_name = payload["workspace_name"]
    observer = payload["observer"]
    observed = payload["observed"]

    logger.info(
        f"Processing hypothesis generation for {observer} observing {observed}"
    )

    # Initialize agent with default config (could be customized per workspace)
    agent = AbducerAgent(db_session, config=AbducerConfig())

    # Execute agent
    result = await agent.execute({
        "workspace_name": workspace_name,
        "observer": observer,
        "observed": observed,
    })

    # Enqueue prediction testing for each new hypothesis
    for hypothesis_id in result.get("hypothesis_ids", []):
        await enqueue_prediction_testing(
            workspace_name=workspace_name,
            hypothesis_id=hypothesis_id,
            parent_task_id=task_id,
        )

    logger.info(
        "Hypothesis generation complete: %d hypotheses, %d premises analyzed",
        result["hypotheses_created"],
        result["premises_analyzed"],
    )

    return result


async def process_prediction_testing(
    db_session: AsyncSession,
    payload: dict[str, Any],
    task_id: int,
) -> dict[str, Any]:
    """
    Process a prediction testing task using the Predictor agent.

    Args:
        db_session: Database session
        payload: Task payload containing workspace_name, hypothesis_id
        task_id: ID of the current task

    Returns:
        Result dictionary with prediction_ids and metrics
    """
    workspace_name = payload["workspace_name"]
    hypothesis_id = payload["hypothesis_id"]
    observer = payload["observer"]
    observed = payload["observed"]

    logger.info(f"Processing prediction testing for hypothesis {hypothesis_id}")

    # Initialize agent
    agent = PredictorAgent(db_session, config=PredictorConfig())

    # Execute agent
    result = await agent.execute({
        "workspace_name": workspace_name,
        "hypothesis_id": hypothesis_id,
        "observer": observer,
        "observed": observed,
    })

    # Enqueue falsification for each new prediction
    for prediction_id in result.get("prediction_ids", []):
        await enqueue_falsification(
            workspace_name=workspace_name,
            prediction_id=prediction_id,
            parent_task_id=task_id,
        )

    logger.info(
        "Prediction testing complete: %d predictions created",
        result["predictions_created"],
    )

    return result


async def process_falsification(
    db_session: AsyncSession,
    payload: dict[str, Any],
    _task_id: int,
) -> dict[str, Any]:
    """
    Process a falsification task using the Falsifier agent.

    Args:
        db_session: Database session
        payload: Task payload containing workspace_name, prediction_id
        task_id: ID of the current task

    Returns:
        Result dictionary with falsification results and metrics
    """
    workspace_name = payload["workspace_name"]
    prediction_id = payload["prediction_id"]
    observer = payload["observer"]
    observed = payload["observed"]

    logger.info(f"Processing falsification for prediction {prediction_id}")

    # Initialize agent
    agent = FalsifierAgent(db_session, config=FalsifierConfig())

    # Execute agent
    result = await agent.execute({
        "workspace_name": workspace_name,
        "prediction_id": prediction_id,
        "observer": observer,
        "observed": observed,
    })

    # If prediction was unfalsified, check if we should trigger induction
    if result.get("final_status") == "unfalsified":
        # Trigger induction task (will check if threshold is met)
        await enqueue_induction(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_unfalsified_count=5,  # Could be configurable
        )

    logger.info(
        "Falsification complete: prediction %s -> %s",
        prediction_id,
        result.get("final_status"),
    )

    return result


async def process_induction(
    db_session: AsyncSession,
    payload: dict[str, Any],
    _task_id: int,
) -> dict[str, Any]:
    """
    Process an induction task using the Inductor agent.

    Args:
        db_session: Database session
        payload: Task payload containing workspace_name, observer, observed
        task_id: ID of the current task

    Returns:
        Result dictionary with induction_ids and metrics
    """
    workspace_name = payload["workspace_name"]
    observer = payload["observer"]
    observed = payload["observed"]

    logger.info(f"Processing induction for {observer} observing {observed}")

    # Initialize agent
    agent = InductorAgent(db_session, config=InductorConfig())

    # Execute agent
    result = await agent.execute({
        "workspace_name": workspace_name,
        "observer": observer,
        "observed": observed,
    })

    logger.info(
        "Induction complete: %d patterns extracted, %d predictions analyzed",
        result["inductions_created"],
        result["predictions_analyzed"],
    )

    return result


# Task processor registry
TASK_PROCESSORS = {
    "hypothesis_generation": process_hypothesis_generation,
    "prediction_testing": process_prediction_testing,
    "falsification": process_falsification,
    "induction": process_induction,
}


async def process_topdown_task(
    db_session: AsyncSession,
    task_type: str,
    payload: dict[str, Any],
    task_id: int,
) -> dict[str, Any]:
    """
    Route a top-down reasoning task to the appropriate processor.

    Args:
        db_session: Database session
        task_type: Type of task to process
        payload: Task payload
        task_id: ID of the current task

    Returns:
        Result dictionary from the processor

    Raises:
        ValueError: If task_type is not recognized
    """
    processor = TASK_PROCESSORS.get(task_type)

    if not processor:
        raise ValueError(f"Unknown top-down task type: {task_type}")

    return await processor(db_session, payload, task_id)
