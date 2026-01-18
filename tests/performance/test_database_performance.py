"""Performance benchmarks for database operations.

Tests measure query execution time for CRUD operations on
reasoning artifacts (hypotheses, predictions, traces, inductions).
"""

import time

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from tests.utils.reasoning_test_helpers import (
    create_test_hypothesis,
    create_test_observations,
    create_test_peer,
    create_test_prediction,
    create_test_workspace,
)


class TestHypothesisCRUDPerformance:
    """Performance benchmarks for Hypothesis CRUD operations."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hypothesis_creation_single(self, db_session: AsyncSession):
        """Benchmark single hypothesis creation.

        Target: < 50ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=5
        )

        hypothesis_data = schemas.HypothesisCreate(
            content="User prefers concise responses",
            observer=observer,
            observed=observed,
            confidence=0.8,
            status="active",
            source_premise_ids=[obs.id for obs in observations],
        )

        # Benchmark
        start_time = time.perf_counter()

        await crud.hypothesis.create_hypothesis(
            db_session,
            workspace_name=workspace_name,
            hypothesis=hypothesis_data,
        )

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert execution_time < 50.0, f"Execution took {execution_time:.2f}ms (target: < 50ms)"

        print(f"\n⏱️  Hypothesis creation: {execution_time:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hypothesis_creation_batch(self, db_session: AsyncSession):
        """Benchmark batch hypothesis creation.

        Target: < 200ms for 10 hypotheses
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=5
        )

        hypotheses_data = [
            schemas.HypothesisCreate(
                content=f"Hypothesis {i+1}",
                observer=observer,
                observed=observed,
                confidence=0.8,
                status="active",
                source_premise_ids=[obs.id for obs in observations],
            )
            for i in range(10)
        ]

        # Benchmark
        start_time = time.perf_counter()

        for hypothesis_data in hypotheses_data:
            await crud.hypothesis.create_hypothesis(
                db_session,
                workspace_name=workspace_name,
                hypothesis=hypothesis_data,
            )

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert execution_time < 200.0, f"Execution took {execution_time:.2f}ms (target: < 200ms)"

        print(f"\n⏱️  Batch hypothesis creation (10): {execution_time:.2f}ms")
        print(f"📊 Avg per hypothesis: {execution_time/10:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hypothesis_query_by_observer(self, db_session: AsyncSession):
        """Benchmark hypothesis query by observer/observed.

        Target: < 100ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create 20 hypotheses
        for i in range(20):
            await create_test_hypothesis(
                db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=f"Hypothesis {i+1}",
                confidence_score=0.8,
            )

        # Benchmark
        start_time = time.perf_counter()

        stmt = select(models.Hypothesis).where(
            models.Hypothesis.workspace_name == workspace_name,
            models.Hypothesis.observer == observer,
            models.Hypothesis.observed == observed,
        )
        result = await db_session.execute(stmt)
        hypotheses = list(result.scalars().all())

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert len(hypotheses) == 20
        assert execution_time < 100.0, f"Execution took {execution_time:.2f}ms (target: < 100ms)"

        print(f"\n⏱️  Hypothesis query (20 records): {execution_time:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hypothesis_count_query(self, db_session: AsyncSession):
        """Benchmark hypothesis count query.

        Target: < 50ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create 50 hypotheses
        for i in range(50):
            await create_test_hypothesis(
                db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=f"Hypothesis {i+1}",
                confidence_score=0.8,
            )

        # Benchmark
        start_time = time.perf_counter()

        stmt = (
            select(func.count())
            .select_from(models.Hypothesis)
            .where(
                models.Hypothesis.workspace_name == workspace_name,
                models.Hypothesis.observer == observer,
                models.Hypothesis.observed == observed,
            )
        )
        result = await db_session.execute(stmt)
        count = result.scalar()

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert count == 50
        assert execution_time < 50.0, f"Execution took {execution_time:.2f}ms (target: < 50ms)"

        print(f"\n⏱️  Hypothesis count (50 records): {execution_time:.2f}ms")


class TestPredictionCRUDPerformance:
    """Performance benchmarks for Prediction CRUD operations."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_prediction_creation_batch(self, db_session: AsyncSession):
        """Benchmark batch prediction creation.

        Target: < 200ms for 10 predictions
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        predictions_data = [
            schemas.PredictionCreate(
                content=f"Prediction {i+1}",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            )
            for i in range(10)
        ]

        # Benchmark
        start_time = time.perf_counter()

        for prediction_data in predictions_data:
            await crud.prediction.create_prediction(
                db_session,
                prediction_data,
                workspace_name,
            )

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert execution_time < 200.0, f"Execution took {execution_time:.2f}ms (target: < 200ms)"

        print(f"\n⏱️  Batch prediction creation (10): {execution_time:.2f}ms")
        print(f"📊 Avg per prediction: {execution_time/10:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_prediction_query_by_hypothesis(self, db_session: AsyncSession):
        """Benchmark prediction query by hypothesis.

        Target: < 100ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        # Create 20 predictions
        for i in range(20):
            await create_test_prediction(
                db_session,
                workspace_name=workspace_name,
                hypothesis_id=hypothesis.id,
                content=f"Prediction {i+1}",
                confidence_score=0.7,
            )

        # Benchmark
        start_time = time.perf_counter()

        stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id == hypothesis.id,
        )
        result = await db_session.execute(stmt)
        predictions = list(result.scalars().all())

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert len(predictions) == 20
        assert execution_time < 100.0, f"Execution took {execution_time:.2f}ms (target: < 100ms)"

        print(f"\n⏱️  Prediction query (20 records): {execution_time:.2f}ms")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_prediction_status_update(self, db_session: AsyncSession):
        """Benchmark prediction status update.

        Target: < 50ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        prediction = await create_test_prediction(
            db_session,
            workspace_name=workspace_name,
            hypothesis_id=hypothesis.id,
            content="User will prefer brief answers",
            confidence_score=0.7,
        )

        # Benchmark
        start_time = time.perf_counter()

        update_data = schemas.PredictionUpdate(status="unfalsified")
        await crud.prediction.update_prediction(
            db_session,
            workspace_name=workspace_name,
            prediction_id=prediction.id,
            prediction=update_data,
        )

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert execution_time < 50.0, f"Execution took {execution_time:.2f}ms (target: < 50ms)"

        print(f"\n⏱️  Prediction status update: {execution_time:.2f}ms")


class TestComplexQueryPerformance:
    """Performance benchmarks for complex queries."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hypothesis_with_predictions_join(self, db_session: AsyncSession):
        """Benchmark hypothesis query with prediction counts.

        Target: < 150ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create 10 hypotheses with 5 predictions each
        for i in range(10):
            hypothesis = await create_test_hypothesis(
                db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=f"Hypothesis {i+1}",
                confidence_score=0.8,
            )

            for j in range(5):
                await create_test_prediction(
                    db_session,
                    workspace_name=workspace_name,
                    hypothesis_id=hypothesis.id,
                    content=f"Prediction {j+1}",
                    confidence_score=0.7,
                )

        # Benchmark
        start_time = time.perf_counter()

        stmt = (
            select(
                models.Hypothesis,
                func.count(models.Prediction.id).label("prediction_count"),
            )
            .outerjoin(models.Prediction)
            .where(
                models.Hypothesis.workspace_name == workspace_name,
                models.Hypothesis.observer == observer,
                models.Hypothesis.observed == observed,
            )
            .group_by(models.Hypothesis.id)
        )
        result = await db_session.execute(stmt)
        rows = result.all()

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert len(rows) == 10
        assert all(row[1] == 5 for row in rows)  # Each hypothesis has 5 predictions
        assert execution_time < 150.0, f"Execution took {execution_time:.2f}ms (target: < 150ms)"

        print(f"\n⏱️  Hypothesis with prediction counts: {execution_time:.2f}ms")
        print(f"📊 Hypotheses: {len(rows)}, Total predictions: {sum(row[1] for row in rows)}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_unfalsified_predictions_query(self, db_session: AsyncSession):
        """Benchmark query for unfalsified predictions.

        Target: < 100ms p95
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        # Create 30 predictions with mixed statuses
        for i in range(30):
            status = "unfalsified" if i % 3 == 0 else "untested"
            await create_test_prediction(
                db_session,
                workspace_name=workspace_name,
                hypothesis_id=hypothesis.id,
                content=f"Prediction {i+1}",
                confidence_score=0.7,
                status=status,
            )

        # Benchmark
        start_time = time.perf_counter()

        stmt = select(models.Prediction).where(
            models.Prediction.workspace_name == workspace_name,
            models.Prediction.hypothesis_id == hypothesis.id,
            models.Prediction.status == "unfalsified",
        )
        result = await db_session.execute(stmt)
        unfalsified = list(result.scalars().all())

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Assertions
        assert len(unfalsified) == 10  # Every 3rd prediction is unfalsified
        assert execution_time < 100.0, f"Execution took {execution_time:.2f}ms (target: < 100ms)"

        print(f"\n⏱️  Unfalsified predictions query: {execution_time:.2f}ms")
        print(f"📊 Found {len(unfalsified)} unfalsified out of 30 total")
