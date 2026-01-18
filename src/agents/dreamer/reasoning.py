"""Reasoning Dream Orchestrator.

This module orchestrates the top-down reasoning workflow during dreams:
1. Check accumulated observations since last reasoning dream
2. If sufficient observations → Run Abducer to generate hypotheses
3. For each new hypothesis → Run Predictor to generate predictions
4. For each new prediction → Run Falsifier to test against evidence
5. If sufficient unfalsified predictions → Run Inductor to extract patterns
6. Run hypothesis retesting for under-tested hypotheses
"""

import logging
from datetime import datetime, timezone

import sentry_sdk
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.agents.abducer.agent import AbducerAgent
from src.agents.falsifier.agent import FalsifierAgent
from src.agents.inductor.agent import InductorAgent
from src.agents.predictor.agent import PredictorAgent
from src.agents.topdown.enqueue import enqueue_undertested_hypothesis_retesting
from src.config import settings

logger = logging.getLogger(__name__)


class ReasoningDreamMetrics:
    """Metrics collected during a reasoning dream."""

    def __init__(
        self,
        workspace_name: str = "",
        observer: str = "",
        observed: str = "",
    ):
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed
        self.hypotheses_generated: int = 0
        self.predictions_generated: int = 0
        self.predictions_falsified: int = 0
        self.predictions_unfalsified: int = 0
        self.inductions_created: int = 0
        self.hypotheses_retested: int = 0
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: datetime | None = None

    def finish(self) -> None:
        """Mark the dream as finished."""
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, int | float]:
        """Convert metrics to dictionary for logging."""
        return {
            "hypotheses_generated": self.hypotheses_generated,
            "predictions_generated": self.predictions_generated,
            "predictions_falsified": self.predictions_falsified,
            "predictions_unfalsified": self.predictions_unfalsified,
            "inductions_created": self.inductions_created,
            "hypotheses_retested": self.hypotheses_retested,
            "duration_seconds": self.duration_seconds,
        }


@sentry_sdk.trace
async def process_reasoning_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    min_observations_threshold: int = 5,
    min_unfalsified_threshold: int = 5,
    max_iterations: int = 10,
) -> ReasoningDreamMetrics:
    """
    Process a reasoning dream: orchestrate hypothesis generation, prediction, falsification, and induction.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer: Name of the observer peer
        observed: Name of the observed peer
        min_observations_threshold: Minimum observations needed to trigger hypothesis generation
        min_unfalsified_threshold: Minimum unfalsified predictions needed to trigger induction
        max_iterations: Maximum number of reasoning iterations per dream

    Returns:
        ReasoningDreamMetrics: Metrics about the reasoning dream execution
    """
    logger.info(
        f"""
🧠💭 ═══════════════════════════════════════
   REASONING DREAM
   Workspace: {workspace_name}
   Observer: {observer}
   Observed: {observed}
═══════════════════════════════════════ 💭🧠"""
    )

    metrics = ReasoningDreamMetrics(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )

    try:
        # Step 1: Check if we have enough observations for hypothesis generation
        observation_count = await _count_recent_observations(
            db, workspace_name, observer, observed
        )

        logger.info(
            f"Found {observation_count} observations (threshold: {min_observations_threshold})"
        )

        if observation_count < min_observations_threshold:
            logger.info(
                f"Not enough observations for reasoning dream. Skipping."
            )
            metrics.finish()
            return metrics

        # Step 2: Run Abducer to generate hypotheses
        logger.info("🔍 Running Abducer to generate hypotheses...")

        # Get count of hypotheses before abducer runs
        stmt_before = select(func.count()).select_from(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result_before = await db.execute(stmt_before)
        count_before = result_before.scalar() or 0

        abducer = AbducerAgent(db=db)
        abducer_result = await abducer.execute(
            {
                "workspace_name": workspace_name,
                "observer": observer,
                "observed": observed,
            }
        )

        # Get count of hypotheses after abducer runs to determine how many were actually created
        stmt_after = select(func.count()).select_from(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result_after = await db.execute(stmt_after)
        count_after = result_after.scalar() or 0
        metrics.hypotheses_generated = count_after - count_before

        logger.info(f"✓ Generated {metrics.hypotheses_generated} hypotheses")

        # Step 3: For each new hypothesis, run Predictor
        # Query actual hypotheses from database instead of relying on agent result
        stmt_hypotheses = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
            models.Hypothesis.status == "active",
        ).order_by(models.Hypothesis.created_at.desc())
        result_hypotheses = await db.execute(stmt_hypotheses)
        hypotheses = list(result_hypotheses.scalars().all())
        hypothesis_ids = [h.id for h in hypotheses[:max_iterations]]
        predictor = PredictorAgent(db=db)

        for hypothesis_id in hypothesis_ids[:max_iterations]:
            logger.info(f"🎯 Running Predictor for hypothesis {hypothesis_id}...")

            # Count predictions before predictor runs
            stmt_pred_before = select(func.count()).select_from(models.Prediction).where(
                models.Prediction.workspace_name == workspace_name,
                models.Prediction.hypothesis_id == hypothesis_id,
            )
            result_pred_before = await db.execute(stmt_pred_before)
            pred_count_before = result_pred_before.scalar() or 0

            predictor_result = await predictor.execute(
                {
                    "workspace_name": workspace_name,
                    "hypothesis_id": hypothesis_id,
                    "observer": observer,
                    "observed": observed,
                }
            )

            # Count predictions after predictor runs
            stmt_pred_after = select(func.count()).select_from(models.Prediction).where(
                models.Prediction.workspace_name == workspace_name,
                models.Prediction.hypothesis_id == hypothesis_id,
            )
            result_pred_after = await db.execute(stmt_pred_after)
            pred_count_after = result_pred_after.scalar() or 0
            predictions_count = pred_count_after - pred_count_before
            metrics.predictions_generated += predictions_count
            logger.info(f"✓ Generated {predictions_count} predictions")

            # Step 4: For each new prediction, run Falsifier
            # Query actual predictions from database
            stmt_predictions = select(models.Prediction).where(
                models.Prediction.workspace_name == workspace_name,
                models.Prediction.hypothesis_id == hypothesis_id,
                models.Prediction.status == "untested",
            ).order_by(models.Prediction.created_at.desc())
            result_predictions = await db.execute(stmt_predictions)
            predictions = list(result_predictions.scalars().all())
            prediction_ids = [p.id for p in predictions]
            falsifier = FalsifierAgent(db=db)

            for prediction_id in prediction_ids:
                logger.info(
                    f"🔬 Running Falsifier for prediction {prediction_id}..."
                )
                falsifier_result = await falsifier.execute(
                    {
                        "workspace_name": workspace_name,
                        "prediction_id": prediction_id,
                        "observer": observer,
                        "observed": observed,
                    }
                )

                # Query the actual prediction status from database
                stmt_pred_status = select(models.Prediction).where(
                    models.Prediction.id == prediction_id,
                    models.Prediction.workspace_name == workspace_name,
                )
                result_pred_status = await db.execute(stmt_pred_status)
                prediction = result_pred_status.scalar_one_or_none()

                if prediction:
                    status = prediction.status
                    if status == "falsified":
                        metrics.predictions_falsified += 1
                    elif status == "unfalsified":
                        metrics.predictions_unfalsified += 1
                    logger.info(f"✓ Prediction status: {status}")
                else:
                    logger.warning(f"Prediction {prediction_id} not found after falsification")

        # Step 5: Check if we have enough unfalsified predictions for induction
        unfalsified_count = await _count_unfalsified_predictions(
            db, workspace_name, observer, observed
        )

        logger.info(
            f"Found {unfalsified_count} unfalsified predictions (threshold: {min_unfalsified_threshold})"
        )

        if unfalsified_count >= min_unfalsified_threshold:
            logger.info("🔗 Running Inductor to extract patterns...")

            # Count inductions before inductor runs
            stmt_ind_before = select(func.count()).select_from(models.Induction).where(
                models.Induction.workspace_name == workspace_name,
                models.Induction.observer == observer,
                models.Induction.observed == observed,
            )
            result_ind_before = await db.execute(stmt_ind_before)
            ind_count_before = result_ind_before.scalar() or 0

            inductor = InductorAgent(db=db)
            inductor_result = await inductor.execute(
                {
                    "workspace_name": workspace_name,
                    "observer": observer,
                    "observed": observed,
                }
            )

            # Count inductions after inductor runs
            stmt_ind_after = select(func.count()).select_from(models.Induction).where(
                models.Induction.workspace_name == workspace_name,
                models.Induction.observer == observer,
                models.Induction.observed == observed,
            )
            result_ind_after = await db.execute(stmt_ind_after)
            ind_count_after = result_ind_after.scalar() or 0
            metrics.inductions_created = ind_count_after - ind_count_before

            logger.info(f"✓ Extracted {metrics.inductions_created} patterns")

        # Step 6: Run hypothesis retesting
        logger.info("🔄 Checking for under-tested hypotheses...")
        retest_task_ids = await enqueue_undertested_hypothesis_retesting(
            workspace_name=workspace_name,
            min_predictions_per_hypothesis=3,
        )
        metrics.hypotheses_retested = len(retest_task_ids)
        logger.info(f"✓ Enqueued {metrics.hypotheses_retested} hypothesis retests")

        metrics.finish()

        logger.info(
            f"""
🧠💭 ═══════════════════════════════════════
   REASONING DREAM COMPLETE
   Duration: {metrics.duration_seconds:.2f}s
   Hypotheses: {metrics.hypotheses_generated}
   Predictions: {metrics.predictions_generated}
   Falsified: {metrics.predictions_falsified}
   Unfalsified: {metrics.predictions_unfalsified}
   Inductions: {metrics.inductions_created}
   Retested: {metrics.hypotheses_retested}
═══════════════════════════════════════ 💭🧠"""
        )

        return metrics

    except Exception as e:
        logger.exception("Error during reasoning dream execution")
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        metrics.finish()
        return metrics


async def _count_recent_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> int:
    """Count recent explicit observations (premises) for hypothesis generation."""
    stmt = (
        select(func.count(models.Document.id))
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.level == "explicit",
        )
    )

    result = await db.execute(stmt)
    return result.scalar_one()


async def _count_unfalsified_predictions(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> int:
    """Count unfalsified predictions for induction."""
    # Get hypothesis IDs for this observer-observed pair
    hypothesis_ids_stmt = select(models.Hypothesis.id).where(
        models.Hypothesis.workspace_name == workspace_name,
        models.Hypothesis.observer == observer,
        models.Hypothesis.observed == observed,
        models.Hypothesis.status == "active",
    )

    hypothesis_ids_result = await db.execute(hypothesis_ids_stmt)
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

    result = await db.execute(stmt)
    return result.scalar_one()
