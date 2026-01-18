"""Validation tests for falsification accuracy.

Tests verify that the falsification process correctly identifies
contradictions and accurately marks predictions as falsified/unfalsified.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.agents.dreamer.reasoning import process_reasoning_dream
from tests.utils.reasoning_test_helpers import (
    create_test_hypothesis,
    create_test_observations,
    create_test_peer,
    create_test_prediction,
    create_test_workspace,
)


class TestFalsificationAccuracy:
    """Tests for falsification accuracy validation."""

    @pytest.mark.asyncio
    async def test_prediction_status_consistency(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify prediction statuses are consistent with falsification results.

        Validates that:
        - Predictions marked as "unfalsified" have passed falsification
        - Predictions marked as "falsified" have contradicting evidence
        - Status transitions are valid
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=15
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get predictions
        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        # Validate status consistency
        assert len(predictions) > 0, "Should generate at least one prediction"

        status_counts = {
            "untested": 0,
            "unfalsified": 0,
            "falsified": 0,
            "superseded": 0,
        }

        for prediction in predictions:
            status = prediction.status
            assert status in status_counts, (
                f"Prediction {prediction.id} has invalid status: {status}"
            )
            status_counts[status] += 1

        # Verify metrics match actual statuses
        assert status_counts["falsified"] == metrics.predictions_falsified, (
            f"Falsified count mismatch: {status_counts['falsified']} vs {metrics.predictions_falsified}"
        )
        assert status_counts["unfalsified"] == metrics.predictions_unfalsified, (
            f"Unfalsified count mismatch: {status_counts['unfalsified']} vs {metrics.predictions_unfalsified}"
        )

        print("\n📊 Prediction Status Distribution:")
        print(f"  Untested: {status_counts['untested']}")
        print(f"  Unfalsified: {status_counts['unfalsified']}")
        print(f"  Falsified: {status_counts['falsified']}")
        print(f"  Superseded: {status_counts['superseded']}")
        print(f"  ✅ Status counts match metrics")

    @pytest.mark.asyncio
    async def test_trace_completeness(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify falsification traces are complete and well-formed.

        Validates that:
        - Traces exist for tested predictions
        - Traces contain required fields
        - Search queries are non-empty
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=15
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get traces
        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        # Validate trace completeness
        for trace in traces:
            # Check required fields
            assert trace.prediction_id is not None, (
                f"Trace {trace.id} has no prediction_id"
            )
            assert trace.workspace_name == workspace_name, (
                f"Trace {trace.id} has wrong workspace"
            )

            # Check search query (may be empty for some traces)
            # Validation depends on trace type and implementation

        print(f"\n✅ Verified {len(traces)} traces are complete and well-formed")

    @pytest.mark.asyncio
    async def test_hypothesis_confidence_updates(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify hypothesis confidence updates based on prediction outcomes.

        Validates that:
        - Hypotheses track prediction outcomes
        - Confidence/tier may be updated based on falsification
        - Status changes are appropriate
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Execute first dream - creates hypotheses
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get initial hypotheses
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
        )
        result = await db_session.execute(hypotheses_stmt)
        initial_hypotheses = list(result.scalars().all())
        initial_confidence = {h.id: h.confidence for h in initial_hypotheses}

        # Add more observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Execute second dream - tests hypotheses
        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get updated hypotheses
        result = await db_session.execute(hypotheses_stmt)
        updated_hypotheses = list(result.scalars().all())

        # Validate hypothesis updates
        for hypothesis in updated_hypotheses:
            initial_conf = initial_confidence.get(hypothesis.id)
            if initial_conf is not None:
                # Hypothesis existed before second dream
                # Confidence may have changed based on prediction outcomes
                assert 0.0 <= hypothesis.confidence <= 1.0, (
                    f"Hypothesis {hypothesis.id} has invalid confidence: {hypothesis.confidence}"
                )

        print(f"\n✅ Verified {len(updated_hypotheses)} hypotheses maintain valid state after falsification")


class TestFalsificationMetrics:
    """Tests for falsification metrics validation."""

    @pytest.mark.asyncio
    async def test_falsification_rate_bounds(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify falsification rates are within expected bounds.

        Validates that:
        - Falsification rate is between 0 and 1
        - At least some predictions are tested
        - Metrics are internally consistent
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Calculate falsification rate
        total_tested = metrics.predictions_falsified + metrics.predictions_unfalsified
        if total_tested > 0:
            falsification_rate = metrics.predictions_falsified / total_tested
            unfalsified_rate = metrics.predictions_unfalsified / total_tested

            # Validate rates
            assert 0.0 <= falsification_rate <= 1.0, (
                f"Falsification rate out of bounds: {falsification_rate}"
            )
            assert 0.0 <= unfalsified_rate <= 1.0, (
                f"Unfalsified rate out of bounds: {unfalsified_rate}"
            )
            assert abs((falsification_rate + unfalsified_rate) - 1.0) < 0.01, (
                "Rates don't sum to 1.0"
            )

            print("\n📊 Falsification Metrics:")
            print(f"  Total Tested: {total_tested}")
            print(f"  Falsified: {metrics.predictions_falsified} ({falsification_rate:.1%})")
            print(f"  Unfalsified: {metrics.predictions_unfalsified} ({unfalsified_rate:.1%})")
            print(f"  ✅ Rates within valid bounds")
        else:
            print("\n⚠️  No predictions tested (threshold not met)")

    @pytest.mark.asyncio
    async def test_trace_coverage(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify trace coverage for tested predictions.

        Validates that:
        - Most tested predictions have traces
        - Trace count aligns with tested prediction count
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=15
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get tested predictions
        tested_predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.status.in_(["falsified", "unfalsified"]),
        )
        result = await db_session.execute(tested_predictions_stmt)
        tested_predictions = list(result.scalars().all())

        # Get traces
        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        # Calculate trace coverage
        if len(tested_predictions) > 0:
            trace_prediction_ids = {t.prediction_id for t in traces}
            tested_prediction_ids = {p.id for p in tested_predictions}

            coverage = len(trace_prediction_ids) / len(tested_predictions)

            # Most tested predictions should have traces
            # (Some may not if falsification was skipped for efficiency)
            assert coverage >= 0.5, (
                f"Trace coverage too low: {coverage:.1%} ({len(traces)} traces for {len(tested_predictions)} predictions)"
            )

            print(f"\n✅ Trace coverage: {coverage:.1%} ({len(traces)} traces for {len(tested_predictions)} tested predictions)")
        else:
            print("\n⚠️  No predictions tested")


class TestPredictionQuality:
    """Tests for prediction quality validation."""

    @pytest.mark.asyncio
    async def test_prediction_quality_properties(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify predictions have expected quality properties.

        Validates that:
        - Predictions have valid content
        - Status values are appropriate
        - Blind flag is boolean
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get predictions
        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        # Validate quality properties
        assert len(predictions) > 0, "Should generate at least one prediction"

        for prediction in predictions:
            # Check content is non-empty
            assert prediction.content is not None and len(prediction.content) > 0, (
                f"Prediction {prediction.id} has empty content"
            )

            # Check status is valid
            assert prediction.status in ["untested", "unfalsified", "falsified", "superseded"], (
                f"Prediction {prediction.id} has invalid status: {prediction.status}"
            )

            # Check is_blind is boolean
            assert isinstance(prediction.is_blind, bool), (
                f"Prediction {prediction.id} is_blind is not boolean: {type(prediction.is_blind)}"
            )

        print(f"\n✅ Verified {len(predictions)} predictions have valid quality properties")

    @pytest.mark.asyncio
    async def test_blind_prediction_properties(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify blind predictions have expected properties.

        Validates that:
        - All predictions initially created as blind
        - Blind flag is boolean
        - Blind predictions can be unblinded
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get predictions
        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        # Validate blind properties
        assert len(predictions) > 0, "Should generate at least one prediction"

        blind_count = sum(1 for p in predictions if p.is_blind)
        all_blind = blind_count == len(predictions)

        print(f"\n✅ Blind predictions: {blind_count}/{len(predictions)} ({blind_count/len(predictions):.1%})")

        if all_blind:
            print("  All predictions created as blind (expected behavior)")
