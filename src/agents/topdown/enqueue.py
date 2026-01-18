"""Task enqueueing functions for top-down reasoning agents.

This module handles task creation and scheduling for the Abducer, Predictor,
Falsifier, and Inductor agents with proper task dependencies.
"""

import logging

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)


async def enqueue_hypothesis_generation(
    workspace_name: str,
    observer: str,
    observed: str,
    min_premise_count: int = 5,
) -> int | None:
    """
    Enqueue a hypothesis generation task (Abducer agent).

    Triggered when sufficient new observations have accumulated for a peer.

    Args:
        workspace_name: Name of the workspace
        observer: Name of the observer peer (agent)
        observed: Name of the observed peer (user being modeled)
        min_premise_count: Minimum premises needed to trigger hypothesis generation

    Returns:
        Task ID if enqueued, None if conditions not met
    """
    async with tracked_db("hypothesis_generation_enqueue") as db_session:
        try:
            # Check if enough premises exist
            premise_count = await _count_recent_premises(
                db_session, workspace_name, observer, observed
            )

            if premise_count < min_premise_count:
                logger.debug(
                    "Not enough premises for hypothesis generation (%d/%d)",
                    premise_count,
                    min_premise_count,
                )
                return None

            # Create work unit key for idempotency
            work_unit_key = f"hypothesis_gen:{workspace_name}:{observer}:{observed}"

            # Check if already enqueued
            existing = await db_session.scalar(
                select(models.QueueItem).where(
                    models.QueueItem.work_unit_key == work_unit_key,
                    models.QueueItem.processed == False,  # noqa: E712
                )
            )

            if existing:
                logger.debug(
                    f"Hypothesis generation task already enqueued: {work_unit_key}"
                )
                return existing.id

            # Create queue item
            payload = {
                "workspace_name": workspace_name,
                "observer": observer,
                "observed": observed,
            }

            stmt = (
                insert(models.QueueItem)
                .values(
                    workspace_name=workspace_name,
                    work_unit_key=work_unit_key,
                    task_type="hypothesis_generation",
                    payload=payload,
                    processed=False,
                    depends_on_task_ids=None,  # First in chain
                )
                .returning(models.QueueItem.id)
            )

            result = await db_session.execute(stmt)
            task_id = result.scalar_one()
            await db_session.commit()

            logger.info(
                "Enqueued hypothesis generation task %s for %s observing %s",
                task_id,
                observer,
                observed,
            )
            return task_id

        except Exception as e:
            logger.exception("Failed to enqueue hypothesis generation task!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            return None


async def enqueue_prediction_testing(
    workspace_name: str,
    hypothesis_id: str,
    parent_task_id: int | None = None,
) -> int | None:
    """
    Enqueue a prediction testing task (Predictor agent).

    Triggered when a new hypothesis is created.

    Args:
        workspace_name: Name of the workspace
        hypothesis_id: ID of the hypothesis to generate predictions from
        parent_task_id: Optional parent task ID for dependency tracking

    Returns:
        Task ID if enqueued, None if failed
    """
    async with tracked_db("prediction_testing_enqueue") as db_session:
        try:
            # Verify hypothesis exists
            hypothesis = await crud.hypothesis.get_hypothesis(
                db_session, workspace_name, hypothesis_id
            )

            # Create work unit key
            work_unit_key = f"pred_test:{workspace_name}:{hypothesis_id}"

            # Check if already enqueued
            existing = await db_session.scalar(
                select(models.QueueItem).where(
                    models.QueueItem.work_unit_key == work_unit_key,
                    models.QueueItem.processed == False,  # noqa: E712
                )
            )

            if existing:
                logger.debug(f"Prediction testing task already enqueued: {work_unit_key}")
                return existing.id

            # Create queue item
            payload = {
                "workspace_name": workspace_name,
                "hypothesis_id": hypothesis_id,
                "observer": hypothesis.observer,
                "observed": hypothesis.observed,
            }

            depends_on = [parent_task_id] if parent_task_id else None

            stmt = (
                insert(models.QueueItem)
                .values(
                    workspace_name=workspace_name,
                    work_unit_key=work_unit_key,
                    task_type="prediction_testing",
                    payload=payload,
                    processed=False,
                    depends_on_task_ids=depends_on,
                )
                .returning(models.QueueItem.id)
            )

            result = await db_session.execute(stmt)
            task_id = result.scalar_one()
            await db_session.commit()

            logger.info(
                f"Enqueued prediction testing task {task_id} for hypothesis {hypothesis_id}"
            )
            return task_id

        except Exception as e:
            logger.exception("Failed to enqueue prediction testing task!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            return None


async def enqueue_falsification(
    workspace_name: str,
    prediction_id: str,
    parent_task_id: int | None = None,
) -> int | None:
    """
    Enqueue a falsification task (Falsifier agent).

    Triggered when a new prediction is created.

    Args:
        workspace_name: Name of the workspace
        prediction_id: ID of the prediction to test
        parent_task_id: Optional parent task ID for dependency tracking

    Returns:
        Task ID if enqueued, None if failed
    """
    async with tracked_db("falsification_enqueue") as db_session:
        try:
            # Verify prediction exists
            prediction = await crud.prediction.get_prediction(
                db_session, workspace_name, prediction_id
            )

            # Create work unit key
            work_unit_key = f"falsify:{workspace_name}:{prediction_id}"

            # Check if already enqueued
            existing = await db_session.scalar(
                select(models.QueueItem).where(
                    models.QueueItem.work_unit_key == work_unit_key,
                    models.QueueItem.processed == False,  # noqa: E712
                )
            )

            if existing:
                logger.debug(f"Falsification task already enqueued: {work_unit_key}")
                return existing.id

            # Get hypothesis to extract observer/observed
            hypothesis = await crud.hypothesis.get_hypothesis(
                db_session, workspace_name, prediction.hypothesis_id
            )

            # Create queue item
            payload = {
                "workspace_name": workspace_name,
                "prediction_id": prediction_id,
                "hypothesis_id": prediction.hypothesis_id,
                "observer": hypothesis.observer,
                "observed": hypothesis.observed,
            }

            depends_on = [parent_task_id] if parent_task_id else None

            stmt = (
                insert(models.QueueItem)
                .values(
                    workspace_name=workspace_name,
                    work_unit_key=work_unit_key,
                    task_type="falsification",
                    payload=payload,
                    processed=False,
                    depends_on_task_ids=depends_on,
                )
                .returning(models.QueueItem.id)
            )

            result = await db_session.execute(stmt)
            task_id = result.scalar_one()
            await db_session.commit()

            logger.info(
                f"Enqueued falsification task {task_id} for prediction {prediction_id}"
            )
            return task_id

        except Exception as e:
            logger.exception("Failed to enqueue falsification task!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            return None


async def enqueue_induction(
    workspace_name: str,
    observer: str,
    observed: str,
    min_unfalsified_count: int = 5,
) -> int | None:
    """
    Enqueue an induction task (Inductor agent).

    Triggered when sufficient unfalsified predictions have accumulated.

    Args:
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        min_unfalsified_count: Minimum unfalsified predictions needed

    Returns:
        Task ID if enqueued, None if conditions not met
    """
    async with tracked_db("induction_enqueue") as db_session:
        try:
            # Count unfalsified predictions
            unfalsified_count = await _count_unfalsified_predictions(
                db_session, workspace_name, observer, observed
            )

            if unfalsified_count < min_unfalsified_count:
                logger.debug(
                    "Not enough unfalsified predictions for induction (%d/%d)",
                    unfalsified_count,
                    min_unfalsified_count,
                )
                return None

            # Create work unit key
            work_unit_key = f"induction:{workspace_name}:{observer}:{observed}"

            # Check if already enqueued
            existing = await db_session.scalar(
                select(models.QueueItem).where(
                    models.QueueItem.work_unit_key == work_unit_key,
                    models.QueueItem.processed == False,  # noqa: E712
                )
            )

            if existing:
                logger.debug(f"Induction task already enqueued: {work_unit_key}")
                return existing.id

            # Create queue item
            payload = {
                "workspace_name": workspace_name,
                "observer": observer,
                "observed": observed,
            }

            stmt = (
                insert(models.QueueItem)
                .values(
                    workspace_name=workspace_name,
                    work_unit_key=work_unit_key,
                    task_type="induction",
                    payload=payload,
                    processed=False,
                    depends_on_task_ids=None,  # Independent task
                )
                .returning(models.QueueItem.id)
            )

            result = await db_session.execute(stmt)
            task_id = result.scalar_one()
            await db_session.commit()

            logger.info(
                "Enqueued induction task %s for %s observing %s",
                task_id,
                observer,
                observed,
            )
            return task_id

        except Exception as e:
            logger.exception("Failed to enqueue induction task!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            return None


async def _count_recent_premises(
    db_session: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> int:
    """Count recent explicit observations (premises) for hypothesis generation."""
    from sqlalchemy import func

    stmt = (
        select(func.count(models.Document.id))
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.level == "explicit",
        )
    )

    result = await db_session.execute(stmt)
    return result.scalar_one()


async def _count_unfalsified_predictions(
    db_session: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> int:
    """Count unfalsified predictions for induction."""
    from sqlalchemy import func

    # Get hypothesis IDs for this observer-observed pair
    hypothesis_ids_stmt = select(models.Hypothesis.id).where(
        models.Hypothesis.workspace_name == workspace_name,
        models.Hypothesis.observer == observer,
        models.Hypothesis.observed == observed,
        models.Hypothesis.status == "active",
    )

    hypothesis_ids_result = await db_session.execute(hypothesis_ids_stmt)
    hypothesis_ids = [row[0] for row in hypothesis_ids_result.fetchall()]

    if not hypothesis_ids:
        return 0

    # Count unfalsified predictions for these hypotheses
    stmt = (
        select(func.count(models.Prediction.id))
        .where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id.in_(hypothesis_ids),
            models.Prediction.status == "unfalsified",
        )
    )

    result = await db_session.execute(stmt)
    return result.scalar_one()


async def enqueue_undertested_hypothesis_retesting(
    workspace_name: str,
    min_predictions_per_hypothesis: int = 3,
) -> list[int]:
    """
    Enqueue prediction testing for under-tested active hypotheses.

    Triggered periodically to ensure all active hypotheses have sufficient testing.

    Args:
        workspace_name: Name of the workspace
        min_predictions_per_hypothesis: Minimum predictions required per hypothesis

    Returns:
        List of task IDs that were enqueued
    """
    async with tracked_db("undertested_hypothesis_check") as db_session:
        try:
            from sqlalchemy import func

            # Find active hypotheses with too few predictions
            stmt = (
                select(
                    models.Hypothesis.id,
                    models.Hypothesis.observer,
                    models.Hypothesis.observed,
                    func.count(models.Prediction.id).label("prediction_count"),
                )
                .outerjoin(
                    models.Prediction,
                    models.Prediction.hypothesis_id == models.Hypothesis.id,
                )
                .where(
                    models.Hypothesis.workspace_name == workspace_name,
                    models.Hypothesis.status == "active",
                )
                .group_by(
                    models.Hypothesis.id,
                    models.Hypothesis.observer,
                    models.Hypothesis.observed,
                )
                .having(func.count(models.Prediction.id) < min_predictions_per_hypothesis)
            )

            result = await db_session.execute(stmt)
            undertested = result.fetchall()

            if not undertested:
                logger.debug(
                    "No under-tested hypotheses found in workspace %s",
                    workspace_name,
                )
                return []

            task_ids: list[int] = []
            for row in undertested:
                hypothesis_id = row.id
                prediction_count = row.prediction_count

                # Enqueue prediction testing
                task_id = await enqueue_prediction_testing(
                    workspace_name=workspace_name,
                    hypothesis_id=hypothesis_id,
                )

                if task_id:
                    task_ids.append(task_id)
                    logger.info(
                        "Enqueued prediction testing for under-tested hypothesis %s (current predictions: %d, required: %d)",
                        hypothesis_id,
                        prediction_count,
                        min_predictions_per_hypothesis,
                    )

            logger.info(
                "Enqueued %d prediction testing tasks for under-tested hypotheses",
                len(task_ids),
            )
            return task_ids

        except Exception as e:
            logger.exception("Failed to enqueue under-tested hypothesis retesting!")
            if settings.SENTRY.ENABLED:
                import sentry_sdk

                sentry_sdk.capture_exception(e)
            return []
