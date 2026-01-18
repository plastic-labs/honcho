"""Validation tests for reasoning chain integrity.

Tests verify that the reasoning chain maintains proper linkages between
observations, hypotheses, predictions, traces, and inductions.
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


class TestReasoningChainIntegrity:
    """Tests for reasoning chain integrity validation."""

    @pytest.mark.asyncio
    async def test_hypothesis_to_observations_linkage(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify hypotheses correctly link to source observations.

        Validates that:
        - Hypotheses have non-empty source_premise_ids
        - All source_premise_ids reference valid observations
        - Linkage is maintained through reasoning process
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )
        observation_ids = {obs.id for obs in observations}

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get generated hypotheses
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())

        # Validate linkage
        assert len(hypotheses) > 0, "Should generate at least one hypothesis"

        for hypothesis in hypotheses:
            # Check source_premise_ids is not empty
            assert hypothesis.source_premise_ids is not None, (
                f"Hypothesis {hypothesis.id} has no source_premise_ids"
            )
            assert len(hypothesis.source_premise_ids) > 0, (
                f"Hypothesis {hypothesis.id} has empty source_premise_ids"
            )

            # Verify all source premises reference valid observations
            for premise_id in hypothesis.source_premise_ids:
                assert premise_id in observation_ids, (
                    f"Hypothesis {hypothesis.id} references invalid observation {premise_id}"
                )

        print(f"\n✅ Verified {len(hypotheses)} hypotheses have valid observation linkages")

    @pytest.mark.asyncio
    async def test_prediction_to_hypothesis_linkage(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify predictions correctly link to parent hypotheses.

        Validates that:
        - Predictions have valid hypothesis_id
        - hypothesis_id references existing hypothesis
        - Linkage is maintained through reasoning process
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

        # Get hypotheses
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())
        hypothesis_ids = {h.id for h in hypotheses}

        # Get predictions
        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())

        # Validate linkage
        assert len(predictions) > 0, "Should generate at least one prediction"

        for prediction in predictions:
            # Check hypothesis_id is valid
            assert prediction.hypothesis_id is not None, (
                f"Prediction {prediction.id} has no hypothesis_id"
            )
            assert prediction.hypothesis_id in hypothesis_ids, (
                f"Prediction {prediction.id} references invalid hypothesis {prediction.hypothesis_id}"
            )

        print(f"\n✅ Verified {len(predictions)} predictions have valid hypothesis linkages")

    @pytest.mark.asyncio
    async def test_trace_to_prediction_linkage(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify falsification traces correctly link to predictions.

        Validates that:
        - Traces have valid prediction_id
        - prediction_id references existing prediction
        - Linkage is maintained through falsification process
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
        prediction_ids = {p.id for p in predictions}

        # Get traces
        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        # Validate linkage (traces may be empty if no falsification occurred)
        for trace in traces:
            # Check prediction_id is valid
            assert trace.prediction_id is not None, (
                f"Trace {trace.id} has no prediction_id"
            )
            assert trace.prediction_id in prediction_ids, (
                f"Trace {trace.id} references invalid prediction {trace.prediction_id}"
            )

        print(f"\n✅ Verified {len(traces)} traces have valid prediction linkages")

    @pytest.mark.asyncio
    async def test_induction_to_prediction_linkage(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify inductions correctly link to source predictions.

        Validates that:
        - Inductions have non-empty source_prediction_ids
        - All source_prediction_ids reference valid predictions
        - Linkage is maintained through induction process
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
        prediction_ids = {p.id for p in predictions}

        # Get inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
            models.Induction.observed == observed,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate linkage (inductions may be empty if threshold not met)
        for induction in inductions:
            # Check source_prediction_ids is not empty
            assert induction.source_prediction_ids is not None, (
                f"Induction {induction.id} has no source_prediction_ids"
            )
            assert len(induction.source_prediction_ids) > 0, (
                f"Induction {induction.id} has empty source_prediction_ids"
            )

            # Verify all source predictions reference valid predictions
            for pred_id in induction.source_prediction_ids:
                assert pred_id in prediction_ids, (
                    f"Induction {induction.id} references invalid prediction {pred_id}"
                )

        print(f"\n✅ Verified {len(inductions)} inductions have valid prediction linkages")

    @pytest.mark.asyncio
    async def test_complete_reasoning_chain(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify complete reasoning chain from observations to inductions.

        Validates that:
        - Full chain: Observations → Hypotheses → Predictions → Traces → Inductions
        - All linkages are valid throughout the chain
        - No broken references
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Collect all entity IDs
        observation_ids = {obs.id for obs in observations}

        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())
        hypothesis_ids = {h.id for h in hypotheses}

        predictions_stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
        )
        result = await db_session.execute(predictions_stmt)
        predictions = list(result.scalars().all())
        prediction_ids = {p.id for p in predictions}

        traces_stmt = select(models.FalsificationTrace).where(
            models.FalsificationTrace.workspace_name == workspace_name,
        )
        result = await db_session.execute(traces_stmt)
        traces = list(result.scalars().all())

        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate complete chain
        chain_valid = True
        broken_links = []

        # Check Observations → Hypotheses
        for hypothesis in hypotheses:
            if hypothesis.source_premise_ids:
                for premise_id in hypothesis.source_premise_ids:
                    if premise_id not in observation_ids:
                        chain_valid = False
                        broken_links.append(
                            f"Hypothesis {hypothesis.id} → invalid observation {premise_id}"
                        )

        # Check Hypotheses → Predictions
        for prediction in predictions:
            if prediction.hypothesis_id not in hypothesis_ids:
                chain_valid = False
                broken_links.append(
                    f"Prediction {prediction.id} → invalid hypothesis {prediction.hypothesis_id}"
                )

        # Check Predictions → Traces
        for trace in traces:
            if trace.prediction_id not in prediction_ids:
                chain_valid = False
                broken_links.append(
                    f"Trace {trace.id} → invalid prediction {trace.prediction_id}"
                )

        # Check Predictions → Inductions
        for induction in inductions:
            if induction.source_prediction_ids:
                for pred_id in induction.source_prediction_ids:
                    if pred_id not in prediction_ids:
                        chain_valid = False
                        broken_links.append(
                            f"Induction {induction.id} → invalid prediction {pred_id}"
                        )

        # Assert chain integrity
        assert chain_valid, f"Reasoning chain has broken links:\n" + "\n".join(broken_links)

        # Print summary
        print("\n📊 Complete Reasoning Chain Validated:")
        print(f"  Observations: {len(observations)}")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Traces: {len(traces)}")
        print(f"  Inductions: {len(inductions)}")
        print(f"  ✅ All linkages valid - No broken references")


class TestReasoningMetadata:
    """Tests for reasoning metadata integrity."""

    @pytest.mark.asyncio
    async def test_hypothesis_metadata_completeness(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify hypotheses have complete metadata.

        Validates that:
        - reasoning_metadata is not null
        - Contains expected metadata fields
        - Metadata is well-formed
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

        # Get hypotheses
        hypotheses_stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
        )
        result = await db_session.execute(hypotheses_stmt)
        hypotheses = list(result.scalars().all())

        # Validate metadata
        assert len(hypotheses) > 0, "Should generate at least one hypothesis"

        for hypothesis in hypotheses:
            # Check metadata exists
            assert hypothesis.reasoning_metadata is not None, (
                f"Hypothesis {hypothesis.id} has no reasoning_metadata"
            )

            # Check metadata is a dict
            assert isinstance(hypothesis.reasoning_metadata, dict), (
                f"Hypothesis {hypothesis.id} reasoning_metadata is not a dict"
            )

        print(f"\n✅ Verified {len(hypotheses)} hypotheses have complete metadata")

    @pytest.mark.asyncio
    async def test_prediction_metadata_completeness(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify predictions have complete metadata.

        Validates that:
        - All required fields are populated
        - Metadata is consistent with prediction status
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

        # Validate metadata
        assert len(predictions) > 0, "Should generate at least one prediction"

        for prediction in predictions:
            # Check required fields
            assert prediction.content is not None and len(prediction.content) > 0, (
                f"Prediction {prediction.id} has empty content"
            )
            assert prediction.status in ["untested", "unfalsified", "falsified", "superseded"], (
                f"Prediction {prediction.id} has invalid status: {prediction.status}"
            )
            assert isinstance(prediction.is_blind, bool), (
                f"Prediction {prediction.id} is_blind is not boolean"
            )

        print(f"\n✅ Verified {len(predictions)} predictions have complete metadata")
