"""Performance benchmarks for reasoning dream execution.

Tests measure execution time for complete reasoning dream workflows
with varying observation counts and complexity levels.
"""

import asyncio
import time
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.dreamer.reasoning import process_reasoning_dream
from tests.utils.reasoning_test_helpers import (
    create_test_observations,
    create_test_peer,
    create_test_workspace,
)


class TestDreamExecutionPerformance:
    """Performance benchmarks for full reasoning dream execution."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_dream_execution_10_observations(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Benchmark dream execution with 10 observations.

        Target: < 10s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        # Benchmark
        start_time = time.perf_counter()

        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert metrics.hypotheses_generated > 0
        assert execution_time < 10.0, f"Execution took {execution_time:.2f}s (target: < 10s)"

        print(f"\n⏱️  Execution time (10 obs): {execution_time:.2f}s")
        print(f"📊 Hypotheses: {metrics.hypotheses_generated}")
        print(f"📊 Predictions: {metrics.predictions_generated}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_dream_execution_50_observations(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Benchmark dream execution with 50 observations.

        Target: < 15s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=50
        )

        # Benchmark
        start_time = time.perf_counter()

        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert metrics.hypotheses_generated > 0
        assert execution_time < 15.0, f"Execution took {execution_time:.2f}s (target: < 15s)"

        print(f"\n⏱️  Execution time (50 obs): {execution_time:.2f}s")
        print(f"📊 Hypotheses: {metrics.hypotheses_generated}")
        print(f"📊 Predictions: {metrics.predictions_generated}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_dream_execution_100_observations(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Benchmark dream execution with 100 observations.

        Target: < 20s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=100
        )

        # Benchmark
        start_time = time.perf_counter()

        metrics = await process_reasoning_dream(
            db=db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            min_observations_threshold=5,
        )

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert metrics.hypotheses_generated > 0
        assert execution_time < 20.0, f"Execution took {execution_time:.2f}s (target: < 20s)"

        print(f"\n⏱️  Execution time (100 obs): {execution_time:.2f}s")
        print(f"📊 Hypotheses: {metrics.hypotheses_generated}")
        print(f"📊 Predictions: {metrics.predictions_generated}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_dream_execution_multiple_iterations(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Benchmark dream execution with multiple iterations.

        Simulates a scenario where hypotheses generate predictions,
        which then get falsified, triggering inductions.

        Target: < 25s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        # Benchmark - Run multiple iterations
        start_time = time.perf_counter()

        total_hypotheses = 0
        total_predictions = 0
        iterations = 3

        for i in range(iterations):
            # Add more observations between iterations
            if i > 0:
                await create_test_observations(
                    db_session, workspace_name, observer, observed, count=10
                )

            metrics = await process_reasoning_dream(
                db=db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                min_observations_threshold=5,
                max_iterations=2,
            )

            total_hypotheses += metrics.hypotheses_generated
            total_predictions += metrics.predictions_generated

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert total_hypotheses > 0
        assert execution_time < 25.0, f"Execution took {execution_time:.2f}s (target: < 25s)"

        print(f"\n⏱️  Execution time ({iterations} iterations): {execution_time:.2f}s")
        print(f"📊 Total Hypotheses: {total_hypotheses}")
        print(f"📊 Total Predictions: {total_predictions}")
        print(f"📊 Avg time per iteration: {execution_time/iterations:.2f}s")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_dream_execution_concurrent_pairs(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Benchmark concurrent dream execution for multiple peer pairs.

        Tests system behavior under concurrent load.

        Target: < 30s for 3 concurrent pairs
        """
        # Setup - Create 3 peer pairs
        workspace_name = await create_test_workspace(db_session)

        pairs = []
        for i in range(3):
            observer = await create_test_peer(
                db_session, workspace_name, f"assistant_{i}"
            )
            observed = await create_test_peer(
                db_session, workspace_name, f"user_{i}"
            )

            await create_test_observations(
                db_session, workspace_name, observer, observed, count=20
            )

            pairs.append((observer, observed))

        # Benchmark - Run all pairs concurrently
        start_time = time.perf_counter()

        async def run_dream(observer: str, observed: str) -> Any:
            return await process_reasoning_dream(
                db=db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                min_observations_threshold=5,
            )

        results = await asyncio.gather(*[
            run_dream(observer, observed) for observer, observed in pairs
        ])

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert all(m.hypotheses_generated > 0 for m in results)
        assert execution_time < 30.0, f"Execution took {execution_time:.2f}s (target: < 30s)"

        total_hypotheses = sum(m.hypotheses_generated for m in results)
        total_predictions = sum(m.predictions_generated for m in results)

        print(f"\n⏱️  Concurrent execution time (3 pairs): {execution_time:.2f}s")
        print(f"📊 Total Hypotheses: {total_hypotheses}")
        print(f"📊 Total Predictions: {total_predictions}")
        print(f"📊 Avg time per pair: {execution_time/3:.2f}s")
