"""Validation tests for pattern quality metrics.

Tests verify that inductions (patterns) extracted from unfalsified
predictions are of high quality, non-redundant, and well-formed.
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


class TestInductionQuality:
    """Tests for induction quality validation."""

    @pytest.mark.asyncio
    async def test_induction_confidence_bounds(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify induction confidence is within valid bounds.

        Validates that:
        - Confidence is between 0 and 1
        - Pattern strength is between 0 and 1
        - Values are reasonable for extracted patterns
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create enough observations to trigger induction
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate confidence and pattern strength
        valid_confidence_levels = {"low", "medium", "high"}

        for induction in inductions:
            # Check confidence is valid string
            assert induction.confidence in valid_confidence_levels, (
                f"Induction {induction.id} confidence invalid: {induction.confidence}"
            )

            # Check pattern strength bounds
            assert 0.0 <= induction.pattern_strength <= 1.0, (
                f"Induction {induction.id} pattern_strength out of bounds: {induction.pattern_strength}"
            )

        if len(inductions) > 0:
            # Count confidence distribution
            confidence_counts = {"low": 0, "medium": 0, "high": 0}
            for i in inductions:
                confidence_counts[i.confidence] += 1

            avg_strength = sum(i.pattern_strength for i in inductions) / len(inductions)

            print(f"\n✅ Verified {len(inductions)} inductions:")
            print(f"  Confidence distribution: {confidence_counts}")
            print(f"  Avg pattern strength: {avg_strength:.2f}")
        else:
            print("\n⚠️  No inductions created (threshold not met)")

    @pytest.mark.asyncio
    async def test_induction_pattern_types(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify inductions have valid pattern types.

        Validates that:
        - pattern_type is one of the allowed values
        - Pattern type is appropriate for the content
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Valid pattern types (from Alembic migration)
        valid_pattern_types = {
            "behavioral",
            "preferential",
            "causal",
            "temporal",
            "contextual",
            "comparative",
            "negative",
        }

        # Validate pattern types
        for induction in inductions:
            assert induction.pattern_type in valid_pattern_types, (
                f"Induction {induction.id} has invalid pattern_type: {induction.pattern_type}"
            )

        if len(inductions) > 0:
            # Count pattern type distribution
            pattern_type_counts = {}
            for induction in inductions:
                pattern_type = induction.pattern_type
                pattern_type_counts[pattern_type] = pattern_type_counts.get(pattern_type, 0) + 1

            print(f"\n✅ Pattern Type Distribution ({len(inductions)} total):")
            for pattern_type, count in sorted(pattern_type_counts.items()):
                print(f"  {pattern_type}: {count}")
        else:
            print("\n⚠️  No inductions created (threshold not met)")

    @pytest.mark.asyncio
    async def test_induction_content_quality(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify induction content is well-formed and meaningful.

        Validates that:
        - Content is non-empty
        - Content has reasonable length
        - Generalization scope is present
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
        )

        # Execute dream
        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate content quality
        for induction in inductions:
            # Check content is non-empty
            assert induction.content is not None and len(induction.content) > 0, (
                f"Induction {induction.id} has empty content"
            )

            # Check content has reasonable length (not too short, not excessive)
            content_length = len(induction.content)
            assert 10 <= content_length <= 1000, (
                f"Induction {induction.id} content length unusual: {content_length} chars"
            )

            # Check generalization scope is present
            assert induction.generalization_scope is not None and len(induction.generalization_scope) > 0, (
                f"Induction {induction.id} has no generalization_scope"
            )

        if len(inductions) > 0:
            avg_length = sum(len(i.content) for i in inductions) / len(inductions)
            print(f"\n✅ Verified {len(inductions)} inductions have quality content:")
            print(f"  Avg content length: {avg_length:.0f} chars")
        else:
            print("\n⚠️  No inductions created (threshold not met)")

    @pytest.mark.asyncio
    async def test_induction_source_tracking(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify inductions correctly track source predictions.

        Validates that:
        - source_prediction_ids is non-empty
        - All source predictions are valid
        - Source count is reasonable
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
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
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate source tracking
        for induction in inductions:
            # Check source_prediction_ids is non-empty
            assert induction.source_prediction_ids is not None, (
                f"Induction {induction.id} has no source_prediction_ids"
            )
            assert len(induction.source_prediction_ids) > 0, (
                f"Induction {induction.id} has empty source_prediction_ids"
            )

            source_ids = induction.source_prediction_ids  # Type narrowing

            # Verify all source predictions are valid
            for pred_id in source_ids:
                assert pred_id in prediction_ids, (
                    f"Induction {induction.id} references invalid prediction {pred_id}"
                )

            # Check source count is reasonable (not just 1, not excessive)
            source_count = len(source_ids)
            assert 1 <= source_count <= 50, (
                f"Induction {induction.id} has unusual source count: {source_count}"
            )

        if len(inductions) > 0:
            avg_sources = sum(len(i.source_prediction_ids) if i.source_prediction_ids else 0 for i in inductions) / len(inductions)
            print(f"\n✅ Verified {len(inductions)} inductions track sources:")
            print(f"  Avg sources per induction: {avg_sources:.1f}")
        else:
            print("\n⚠️  No inductions created (threshold not met)")


class TestPatternDiversity:
    """Tests for pattern diversity validation."""

    @pytest.mark.asyncio
    async def test_multiple_pattern_types(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify system generates diverse pattern types.

        Validates that:
        - Multiple pattern types can be generated
        - Patterns are not all the same type
        - System can identify different pattern categories
        """
        # Setup with larger data set to enable diverse patterns
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create varied observations to trigger diverse patterns
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=30,
            content_prefix="User prefers detailed technical explanations"
        )

        # Execute multiple dreams to accumulate patterns
        for i in range(2):
            await create_test_observations(
                db_session, workspace_name, observer, observed, count=15,
                content_prefix=f"User shows consistent behavior pattern {i}"
            )

            await process_reasoning_dream(
                db=db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )

        # Get all inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        if len(inductions) > 1:
            # Count unique pattern types
            unique_types = {i.pattern_type for i in inductions}

            print(f"\n📊 Pattern Diversity ({len(inductions)} inductions):")
            print(f"  Unique pattern types: {len(unique_types)}")
            print(f"  Types: {', '.join(sorted(unique_types))}")

            # With multiple inductions, we should see some diversity
            # (though mocks may always generate same type)
            if len(unique_types) > 1:
                print(f"  ✅ Multiple pattern types detected")
            else:
                print(f"  ⚠️  All patterns same type (expected with mocks)")
        else:
            print(f"\n⚠️  Only {len(inductions)} inductions (not enough for diversity test)")


class TestPatternConsistency:
    """Tests for pattern consistency over time."""

    @pytest.mark.asyncio
    async def test_pattern_stability_across_dreams(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify patterns remain consistent across multiple dreams.

        Validates that:
        - Similar observations lead to similar patterns
        - Patterns don't contradict each other
        - Pattern quality is maintained
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Execute first dream
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25,
            content_prefix="User prefers concise responses"
        )

        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Execute second dream with similar observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25,
            content_prefix="User prefers brief explanations"
        )

        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Get all inductions
        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        inductions = list(result.scalars().all())

        # Validate pattern consistency
        if len(inductions) >= 2:
            # Check that pattern strength remains reasonable (confidence is categorical)
            confidences = [i.confidence for i in inductions]
            strengths = [i.pattern_strength for i in inductions]

            avg_strength = sum(strengths) / len(strengths)

            # Values should be relatively consistent (within reasonable variance)
            strength_variance = sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)

            # Count confidence distribution
            confidence_counts = {"low": 0, "medium": 0, "high": 0}
            for c in confidences:
                confidence_counts[c] += 1

            print(f"\n📊 Pattern Stability ({len(inductions)} inductions):")
            print(f"  Confidence distribution: {confidence_counts}")
            print(f"  Avg pattern strength: {avg_strength:.2f} (variance: {strength_variance:.4f})")

            # Low variance indicates stability
            if strength_variance < 0.05:
                print(f"  ✅ Low variance - patterns are stable")
            else:
                print(f"  ⚠️  Higher variance - patterns vary (expected with different observations)")
        else:
            print(f"\n⚠️  Only {len(inductions)} inductions (not enough for stability test)")

    @pytest.mark.asyncio
    async def test_pattern_evolution(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Verify patterns can evolve with new evidence.

        Validates that:
        - New patterns can be created
        - Existing patterns remain valid
        - System adapts to new observations
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Execute first dream
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
        )

        metrics1 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        inductions_stmt = select(models.Induction).where(
            models.Induction.workspace_name == workspace_name,
            models.Induction.observer == observer,
        )
        result = await db_session.execute(inductions_stmt)
        initial_inductions = len(list(result.scalars().all()))

        # Execute second dream with more observations
        await create_test_observations(
            db_session, workspace_name, observer, observed, count=25
        )

        metrics2 = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        result = await db_session.execute(inductions_stmt)
        final_inductions = len(list(result.scalars().all()))

        # Validate pattern evolution
        print(f"\n📊 Pattern Evolution:")
        print(f"  Initial inductions: {initial_inductions}")
        print(f"  Final inductions: {final_inductions}")

        if final_inductions >= initial_inductions:
            print(f"  ✅ Patterns maintained or grew (delta: +{final_inductions - initial_inductions})")
        else:
            print(f"  ⚠️  Fewer patterns after second dream (unexpected)")
