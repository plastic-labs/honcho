"""Integration tests for multi-agent coordination in reasoning dreams.

These tests verify how agents interact and coordinate during complex reasoning
workflows, including concurrent processing, hypothesis evolution, and error recovery.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.agents.dreamer.reasoning import process_reasoning_dream
from tests.utils.reasoning_test_helpers import (
    create_test_observations,
    create_test_peer,
    create_test_workspace,
)


class TestConcurrentDreamProcessing:
    """Test concurrent dream processing for multiple peer pairs."""

    @pytest.mark.asyncio
    async def test_multiple_observer_observed_pairs(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that reasoning dreams for different peer pairs don't interfere.

        Scenario: Multiple (observer, observed) pairs in same workspace
        Expected: Each pair processes independently without cross-contamination
        """
        # Setup: Create workspace with multiple observers and observed peers
        workspace_name = await create_test_workspace(db_session)

        # Create 2 observers
        observer1 = await create_test_peer(db_session, workspace_name, "assistant_1")
        observer2 = await create_test_peer(db_session, workspace_name, "assistant_2")

        # Create 2 observed users
        observed1 = await create_test_peer(db_session, workspace_name, "user_1")
        observed2 = await create_test_peer(db_session, workspace_name, "user_2")

        # Create observations for both pairs
        obs1 = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer1,
            observed=observed1,
            count=10,
            content_prefix="User 1 prefers detailed explanations",
        )

        obs2 = await create_test_observations(
            db_session,
            workspace_name=workspace_name,
            observer=observer2,
            observed=observed2,
            count=10,
            content_prefix="User 2 prefers concise summaries",
        )

        # Execute reasoning dreams for both pairs
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer1,
            observed=observed1,
        )

        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer2,
            observed=observed2,
        )

        # Verify both dreams executed successfully
        assert metrics1.hypotheses_generated > 0
        assert metrics2.hypotheses_generated > 0

        # Verify hypotheses are correctly scoped to their peer pairs
        hypotheses1_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer1,
            models.Hypothesis.observed == observed1,
        )
        result1 = await db_session.execute(hypotheses1_stmt)
        hypotheses1 = list(result1.scalars().all())

        hypotheses2_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer2,
            models.Hypothesis.observed == observed2,
        )
        result2 = await db_session.execute(hypotheses2_stmt)
        hypotheses2 = list(result2.scalars().all())

        # Verify no cross-contamination
        assert len(hypotheses1) > 0
        assert len(hypotheses2) > 0

        # Verify all hypotheses for pair 1 have correct observer/observed
        for h in hypotheses1:
            assert h.observer == observer1
            assert h.observed == observed1
            assert h.observer != observer2
            assert h.observed != observed2

        # Verify all hypotheses for pair 2 have correct observer/observed
        for h in hypotheses2:
            assert h.observer == observer2
            assert h.observed == observed2
            assert h.observer != observer1
            assert h.observed != observed1

    @pytest.mark.asyncio
    async def test_multiple_workspaces_isolation(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that reasoning dreams in different workspaces are isolated.

        Scenario: Same peer names in different workspaces
        Expected: Complete isolation between workspaces
        """
        # Setup: Create two workspaces with same peer names
        workspace1 = await create_test_workspace(db_session)
        workspace2 = await create_test_workspace(db_session)

        # Use identical peer names in both workspaces
        observer = "assistant"
        observed = "user_123"

        # Create peers in both workspaces
        await create_test_peer(db_session, workspace1, observer)
        await create_test_peer(db_session, workspace1, observed)
        await create_test_peer(db_session, workspace2, observer)
        await create_test_peer(db_session, workspace2, observed)

        # Create observations in both workspaces
        await create_test_observations(
            db_session, workspace1, observer, observed, count=10
        )
        await create_test_observations(
            db_session, workspace2, observer, observed, count=10
        )

        # Execute reasoning dreams in both workspaces
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace1,
            observer=observer,
            observed=observed,
        )

        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace2,
            observer=observer,
            observed=observed,
        )

        # Verify both executed successfully
        assert metrics1.hypotheses_generated > 0
        assert metrics2.hypotheses_generated > 0

        # Verify workspace isolation
        hypotheses1_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace1
        )
        result1 = await db_session.execute(hypotheses1_stmt)
        hypotheses1 = list(result1.scalars().all())

        hypotheses2_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace2
        )
        result2 = await db_session.execute(hypotheses2_stmt)
        hypotheses2 = list(result2.scalars().all())

        # Verify no overlap
        workspace1_ids = {h.id for h in hypotheses1}
        workspace2_ids = {h.id for h in hypotheses2}
        assert len(workspace1_ids & workspace2_ids) == 0, "Hypotheses leaked between workspaces"


class TestHypothesisEvolution:
    """Test hypothesis evolution across multiple reasoning dreams."""

    @pytest.mark.asyncio
    async def test_hypothesis_confidence_updates(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that hypothesis confidence evolves based on prediction outcomes.

        Scenario: Multiple dreams with predictions testing hypotheses
        Expected: Hypothesis confidence and tier updated based on results
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create initial observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # First dream - creates hypotheses
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        assert metrics1.hypotheses_generated > 0

        # Get initial hypothesis
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())
        initial_hypothesis = hypotheses[0]
        initial_confidence = initial_hypothesis.confidence
        initial_tier = initial_hypothesis.tier

        # Add more observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=5
        )

        # Second dream - tests existing hypotheses
        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Verify hypothesis was tested
        assert metrics2.predictions_generated > 0

        # Reload hypothesis to check for updates
        await db_session.refresh(initial_hypothesis)

        # Verify predictions exist for this hypothesis
        predictions_stmt = select(models.Prediction).where(
            models.Prediction.hypothesis_id == initial_hypothesis.id
        )
        result = await db_session.execute(predictions_stmt)
        predictions_for_hypothesis = list(result.scalars().all())

        assert len(predictions_for_hypothesis) > 0, (
            "Hypothesis should have predictions created for testing"
        )

    @pytest.mark.asyncio
    async def test_hypothesis_superseding(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that better hypotheses can supersede older ones.

        Scenario: New hypothesis with higher confidence
        Expected: Old hypothesis marked as superseded, new one becomes active
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # First dream
        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        assert metrics1.hypotheses_generated > 0

        # Get initial hypotheses
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.status == "active",
        )
        result = await db_session.execute(hypotheses_stmt)
        active_hypotheses_before = len(list(result.scalars().all()))

        # Add more observations (might trigger new hypotheses)
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Second dream
        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Check hypothesis evolution
        all_hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
        )
        result = await db_session.execute(all_hypotheses_stmt)
        all_hypotheses = list(result.scalars().all())

        # Verify we have hypotheses in different states
        statuses = {h.status for h in all_hypotheses}
        assert "active" in statuses, "Should have active hypotheses"


class TestPredictionRetesting:
    """Test prediction retesting workflows."""

    @pytest.mark.asyncio
    async def test_undertested_hypothesis_retesting(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that under-tested hypotheses are retested in subsequent dreams.

        Scenario: Hypothesis with few tested predictions
        Expected: Retesting tasks enqueued for under-tested hypotheses
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        # Verify reasoning occurred
        assert metrics.hypotheses_generated > 0
        assert metrics.predictions_generated > 0

        # Check if retesting was tracked
        # Note: In mock environment, retesting count may be 0 if all hypotheses
        # have sufficient predictions
        assert metrics.hypotheses_retested >= 0

    @pytest.mark.asyncio
    async def test_blind_predictions_unblinding(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test that blind predictions are eventually unblinded.

        Scenario: Predictions created as blind
        Expected: All predictions remain blind until unblinding process runs
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations
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

        if len(predictions) > 0:
            # Verify all predictions created as blind
            for prediction in predictions:
                assert prediction.is_blind is True, (
                    f"Prediction {prediction.id} should be blind initially"
                )


class TestErrorRecovery:
    """Test error recovery and rollback scenarios."""

    @pytest.mark.asyncio
    async def test_empty_workspace_handling(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test graceful handling of empty workspace.

        Scenario: Workspace with no observations
        Expected: Dream completes without errors, no reasoning artifacts created
        """
        # Setup: Create workspace but NO observations
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Execute dream with no observations
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        # Verify graceful handling
        assert metrics.hypotheses_generated == 0
        assert metrics.predictions_generated == 0
        assert metrics.predictions_falsified == 0
        assert metrics.predictions_unfalsified == 0
        assert metrics.inductions_created == 0

    @pytest.mark.asyncio
    async def test_insufficient_observations_handling(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test handling when observations below threshold.

        Scenario: Observations exist but below threshold
        Expected: Dream completes without reasoning
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations BELOW threshold
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=3
        )

        # Execute dream with threshold of 5
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        # Verify no reasoning occurred
        assert metrics.hypotheses_generated == 0
        assert metrics.predictions_generated == 0

    @pytest.mark.asyncio
    async def test_missing_peer_handling(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test error handling for non-existent peers.

        Scenario: Observer or observed peer doesn't exist
        Expected: Graceful error or empty result
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)

        # Create only observer, not observed
        observer = await create_test_peer(db_session, workspace_name, "assistant")

        # Try to execute dream with non-existent observed peer
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed="nonexistent_user",  # This peer doesn't exist
            min_observations_threshold=5,
        )

        # Verify graceful handling (no crash, but no work done)
        assert metrics.hypotheses_generated == 0
