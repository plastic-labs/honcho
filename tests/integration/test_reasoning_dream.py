"""Integration tests for reasoning dream workflow.

These tests verify the end-to-end functionality of the reasoning dream system,
including the full cycle from observations to hypotheses, predictions,
falsification, and induction.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.agents.dreamer.reasoning import ReasoningDreamMetrics, process_reasoning_dream
from tests.utils.reasoning_test_helpers import (
    assert_hypothesis_valid,
    assert_induction_valid,
    assert_prediction_valid,
    assert_trace_valid,
    create_test_observations,
    create_test_peer,
    create_test_workspace,
)


class TestReasoningDreamWorkflow:
    """Test the complete reasoning dream workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_dream_workflow(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test complete reasoning cycle: observations → hypotheses → predictions → falsification → induction.

        This is the most critical integration test for Phase 6. It verifies that:
        1. Observations trigger reasoning dreams when threshold is met
        2. Abducer generates hypotheses from observations
        3. Predictor generates predictions from hypotheses
        4. Falsifier tests predictions through contradiction search
        5. Inductor extracts patterns from unfalsified predictions
        6. All reasoning artifacts are correctly linked
        7. Dream metrics are collected accurately
        """
        # Setup: Create workspace and peers
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create enough observations to trigger reasoning (default threshold: 5)
        observations = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            count=10,  # Above threshold
            content_prefix="User prefers concise responses for topic",
        )

        # Verify observations were created
        assert len(observations) == 10
        observation_ids = [obs.id for obs in observations]

        # Execute reasoning dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
            min_unfalsified_threshold=3,
            max_iterations=5,
        )

        # Verify metrics were collected
        assert isinstance(metrics, ReasoningDreamMetrics)
        assert metrics.workspace_name == workspace_name
        assert metrics.observer == observer
        assert metrics.observed == observed

        # Verify hypotheses were generated
        assert metrics.hypotheses_generated > 0, "No hypotheses were generated"

        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())

        assert len(hypotheses) > 0, "No hypotheses found in database"

        # Validate hypothesis properties
        for hypothesis in hypotheses:
            assert_hypothesis_valid(hypothesis, observer, observed)
            # Verify hypothesis is linked to source observations
            assert hypothesis.source_premise_ids is not None and len(hypothesis.source_premise_ids) > 0, "Hypothesis has no source premises"

        # Verify predictions were generated
        assert metrics.predictions_generated > 0, "No predictions were generated"

        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id.in_([h.id for h in hypotheses]),
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        assert len(predictions) > 0, "No predictions found in database"

        # Validate prediction properties
        for prediction in predictions:
            assert_prediction_valid(prediction, prediction.hypothesis_id)
            assert prediction.is_blind is True, "Predictions should be blind"

        # Verify falsification occurred
        total_falsification = metrics.predictions_falsified + metrics.predictions_unfalsified
        assert total_falsification > 0, "No predictions were tested"

        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
            models.FalsificationTrace.prediction_id.in_([p.id for p in predictions]),
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        assert len(traces) > 0, "No falsification traces found"

        # Validate trace properties
        for trace in traces:
            assert_trace_valid(trace, trace.prediction_id)
            # Verify trace has search queries
            assert trace.search_queries is not None and len(trace.search_queries) > 0, "Trace has no search queries"

        # If enough predictions were unfalsified, verify inductions were created
        if metrics.predictions_unfalsified >= 3:
            assert metrics.inductions_created > 0, "No inductions created despite unfalsified predictions"

            inductions_stmt = select(models.Induction).where(
                models.Induction.workspace_name == workspace_name,
                models.Induction.observer == observer,
                models.Induction.observed == observed,
            )
            result = await db_session.execute(inductions_stmt)
            inductions = list(result.scalars().all())

            assert len(inductions) > 0, "No inductions found in database"

            # Validate induction properties
            for induction in inductions:
                assert_induction_valid(induction, observer, observed)
                # Verify induction is linked to source predictions
                assert induction.source_prediction_ids is not None and len(induction.source_prediction_ids) > 0, "Induction has no source predictions"

            # Verify reasoning chain integrity
            # Trace back from induction → prediction → hypothesis → observation
            induction = inductions[0]
            # Get source prediction
            assert induction.source_prediction_ids is not None and len(induction.source_prediction_ids) > 0
            source_prediction_id = induction.source_prediction_ids[0]
            prediction = next(p for p in predictions if p.id == source_prediction_id)
            # Get source hypothesis
            hypothesis = next(h for h in hypotheses if h.id == prediction.hypothesis_id)
            # Verify hypothesis links to observations
            assert hypothesis.source_premise_ids is not None and len(hypothesis.source_premise_ids) > 0
            # Verify at least one source premise is in our observations
            assert any(
                premise_id in observation_ids for premise_id in hypothesis.source_premise_ids
            ), "Hypothesis not linked to test observations"

        # Verify dream execution time
        assert metrics.duration_seconds > 0, "Dream duration not recorded"
        assert metrics.duration_seconds < 60, "Dream took too long (> 60 seconds)"

    @pytest.mark.asyncio
    async def test_reasoning_dream_threshold_enforcement(self, db_session: AsyncSession):
        """Test that reasoning dreams respect observation count thresholds."""
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations BELOW threshold
        observations = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            count=3,  # Below default threshold of 5
        )

        # Execute reasoning dream with default threshold
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,  # Threshold not met
            min_unfalsified_threshold=3,
            max_iterations=5,
        )

        # Verify NO reasoning occurred
        assert metrics.hypotheses_generated == 0, "Hypotheses generated despite threshold not met"
        assert metrics.predictions_generated == 0, "Predictions generated despite no hypotheses"
        assert metrics.predictions_falsified == 0
        assert metrics.predictions_unfalsified == 0
        assert metrics.inductions_created == 0

    @pytest.mark.asyncio
    async def test_reasoning_dream_max_iterations(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that reasoning dreams respect max iteration limits."""
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create many observations
        observations = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            count=50,  # Many observations
        )

        # Execute reasoning dream with LIMITED iterations
        max_iterations = 2
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
            min_unfalsified_threshold=3,
            max_iterations=max_iterations,  # Limit iterations
        )

        # Verify iteration limit was respected
        # Each iteration generates at most a few predictions
        # With max_iterations=2, we should have limited predictions
        assert metrics.predictions_generated <= max_iterations * 10, (
            f"Too many predictions generated ({metrics.predictions_generated}), "
            f"max_iterations may not have been enforced"
        )

    @pytest.mark.asyncio
    async def test_reasoning_dream_metrics_accuracy(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that reasoning dream metrics accurately reflect work performed."""
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations
        observations = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            count=10,
        )

        # Execute reasoning dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
            min_unfalsified_threshold=3,
            max_iterations=5,
        )

        # Count actual database records
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())

        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id.in_([h.id for h in hypotheses]),
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
            models.Induction.observed == observed,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Verify metrics match database counts
        assert metrics.hypotheses_generated == len(hypotheses), (
            f"Hypothesis count mismatch: metrics={metrics.hypotheses_generated}, "
            f"database={len(hypotheses)}"
        )
        assert metrics.predictions_generated == len(predictions), (
            f"Prediction count mismatch: metrics={metrics.predictions_generated}, "
            f"database={len(predictions)}"
        )

        # Count falsified/unfalsified predictions
        falsified_count = sum(1 for p in predictions if p.status == "falsified")
        unfalsified_count = sum(1 for p in predictions if p.status == "unfalsified")

        assert metrics.predictions_falsified == falsified_count, (
            f"Falsified count mismatch: metrics={metrics.predictions_falsified}, "
            f"actual={falsified_count}"
        )
        assert metrics.predictions_unfalsified == unfalsified_count, (
            f"Unfalsified count mismatch: metrics={metrics.predictions_unfalsified}, "
            f"actual={unfalsified_count}"
        )

        assert metrics.inductions_created == len(inductions), (
            f"Induction count mismatch: metrics={metrics.inductions_created}, "
            f"database={len(inductions)}"
        )

    @pytest.mark.asyncio
    async def test_reasoning_dream_idempotency(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that running reasoning dream multiple times doesn't duplicate results."""
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations
        observations = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            count=10,
        )

        # Execute reasoning dream FIRST time
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
            min_unfalsified_threshold=3,
            max_iterations=5,
        )

        # Count records after first run
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses_after_first = len(list(result.scalars().all()))

        # Execute reasoning dream SECOND time (no new observations)
        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
            min_unfalsified_threshold=3,
            max_iterations=5,
        )

        # Count records after second run
        result = await db_session.execute(hypotheses_stmt)
        hypotheses_after_second = len(list(result.scalars().all()))

        # Verify no duplication occurred
        # Second run should generate NO new hypotheses (no new observations)
        assert metrics2.hypotheses_generated == 0, (
            f"Second reasoning dream generated {metrics2.hypotheses_generated} hypotheses, "
            "expected 0 (no new observations)"
        )
        assert hypotheses_after_second == hypotheses_after_first, (
            f"Hypothesis count changed: {hypotheses_after_first} → {hypotheses_after_second}"
        )
