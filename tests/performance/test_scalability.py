"""Scalability benchmarks for reasoning dream workflow.

Tests measure system behavior under increasing load and data volume.
"""

import asyncio
import time

import pytest
from sqlalchemy import func, select
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


class TestDataVolumeScaling:
    """Test system performance with increasing data volumes."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.slow
    async def test_scaling_observations_10_to_100(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test dream execution scaling from 10 to 100 observations.

        Measures how execution time scales with observation count.
        """
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observation_counts = [10, 25, 50, 75, 100]
        results = []

        for count in observation_counts:
            # Create observations
            await create_test_observations(
                db_session, workspace_name, observer, observed, count=count
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

            results.append({
                "observation_count": count,
                "execution_time": execution_time,
                "hypotheses": metrics.hypotheses_generated,
                "predictions": metrics.predictions_generated,
            })

        # Assertions - Check for reasonable scaling
        # Execution time should scale sub-linearly (not exponentially)
        time_10 = results[0]["execution_time"]
        time_100 = results[4]["execution_time"]
        scaling_factor = time_100 / time_10

        assert scaling_factor < 5.0, (
            f"Scaling factor {scaling_factor:.2f}x is too high "
            f"(10x data should take < 5x time)"
        )

        # Output results
        print("\n📊 Observation Scaling Results:")
        print(f"{'Obs Count':<12} {'Time (s)':<10} {'Hypotheses':<12} {'Predictions':<12}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['observation_count']:<12} "
                f"{r['execution_time']:<10.2f} "
                f"{r['hypotheses']:<12} "
                f"{r['predictions']:<12}"
            )
        print(f"\n⚡ Scaling Factor (10→100): {scaling_factor:.2f}x")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_scaling_existing_hypotheses(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test dream execution with increasing number of existing hypotheses.

        Measures impact of existing hypothesis volume on new dream execution.
        """
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        hypothesis_counts = [0, 10, 25, 50]
        results = []

        for count in hypothesis_counts:
            # Create existing hypotheses
            if count > len([h for r in results for h in range(r["hypothesis_count"])]):
                hypotheses_to_create = count - len([h for r in results for h in range(r.get("hypothesis_count", 0))])
                for i in range(hypotheses_to_create):
                    await create_test_hypothesis(
                        db_session,
                        workspace_name=workspace_name,
                        observer=observer,
                        observed=observed,
                        content=f"Existing hypothesis {i+1}",
                        confidence_score=0.8,
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

            results.append({
                "hypothesis_count": count,
                "execution_time": execution_time,
                "new_hypotheses": metrics.hypotheses_generated,
            })

        # Output results
        print("\n📊 Existing Hypothesis Scaling Results:")
        print(f"{'Existing H':<15} {'Time (s)':<10} {'New H':<10}")
        print("-" * 40)
        for r in results:
            print(
                f"{r['hypothesis_count']:<15} "
                f"{r['execution_time']:<10.2f} "
                f"{r['new_hypotheses']:<10}"
            )


class TestConcurrentLoadScaling:
    """Test system performance under concurrent load."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_workspaces_1_to_5(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test concurrent dream execution for 1 to 5 workspaces.

        Measures how system handles increasing concurrent workloads.
        """
        workspace_counts = [1, 2, 3, 5]
        results = []

        for workspace_count in workspace_counts:
            # Setup multiple workspaces
            workspaces = []
            for i in range(workspace_count):
                workspace_name = await create_test_workspace(db_session)
                observer = await create_test_peer(db_session, workspace_name, "assistant")
                observed = await create_test_peer(db_session, workspace_name, "user_123")

                await create_test_observations(
                    db_session, workspace_name, observer, observed, count=20
                )

                workspaces.append((workspace_name, observer, observed))

            # Benchmark concurrent execution
            start_time = time.perf_counter()

            async def run_dream(ws: str, obs_r: str, obs_d: str):
                return await process_reasoning_dream(
                    db=db_session,
                    workspace_name=ws,
                    observer=obs_r,
                    observed=obs_d,
                    min_observations_threshold=5,
                )

            metrics_list = await asyncio.gather(*[
                run_dream(ws, obs_r, obs_d) for ws, obs_r, obs_d in workspaces
            ])

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            total_hypotheses = sum(m.hypotheses_generated for m in metrics_list)

            results.append({
                "workspace_count": workspace_count,
                "execution_time": execution_time,
                "avg_time": execution_time / workspace_count,
                "total_hypotheses": total_hypotheses,
            })

        # Output results
        print("\n📊 Concurrent Workspace Scaling Results:")
        print(f"{'Workspaces':<12} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Total H':<10}")
        print("-" * 55)
        for r in results:
            print(
                f"{r['workspace_count']:<12} "
                f"{r['execution_time']:<15.2f} "
                f"{r['avg_time']:<15.2f} "
                f"{r['total_hypotheses']:<10}"
            )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_peer_pairs_1_to_10(
        self, db_session: AsyncSession, mock_all_reasoning_agents
    ):
        """Test concurrent dream execution for 1 to 10 peer pairs in same workspace.

        Measures workspace-level concurrency handling.
        """
        workspace_name = await create_test_workspace(db_session)

        pair_counts = [1, 3, 5, 10]
        results = []

        for pair_count in pair_counts:
            # Setup multiple peer pairs
            pairs = []
            for i in range(pair_count):
                observer = await create_test_peer(
                    db_session, workspace_name, f"assistant_{i}"
                )
                observed = await create_test_peer(
                    db_session, workspace_name, f"user_{i}"
                )

                await create_test_observations(
                    db_session, workspace_name, observer, observed, count=15
                )

                pairs.append((observer, observed))

            # Benchmark concurrent execution
            start_time = time.perf_counter()

            async def run_dream(obs_r: str, obs_d: str):
                return await process_reasoning_dream(
                    db=db_session,
                    workspace_name=workspace_name,
                    observer=obs_r,
                    observed=obs_d,
                    min_observations_threshold=5,
                )

            metrics_list = await asyncio.gather(*[
                run_dream(obs_r, obs_d) for obs_r, obs_d in pairs
            ])

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            total_hypotheses = sum(m.hypotheses_generated for m in metrics_list)

            results.append({
                "pair_count": pair_count,
                "execution_time": execution_time,
                "avg_time": execution_time / pair_count,
                "total_hypotheses": total_hypotheses,
            })

        # Output results
        print("\n📊 Concurrent Peer Pair Scaling Results:")
        print(f"{'Pairs':<8} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Total H':<10}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['pair_count']:<8} "
                f"{r['execution_time']:<15.2f} "
                f"{r['avg_time']:<15.2f} "
                f"{r['total_hypotheses']:<10}"
            )


class TestMemoryScaling:
    """Test database query performance with large data volumes."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.slow
    async def test_hypothesis_query_100_to_1000(self, db_session: AsyncSession):
        """Test hypothesis queries with 100 to 1000 records.

        Measures query performance degradation with large datasets.
        """
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        record_counts = [100, 250, 500, 1000]
        results = []

        for count in record_counts:
            # Create hypotheses up to count
            current_count_stmt = (
                select(func.count())
                .select_from(models.Hypothesis)
                .where(
                    models.Hypothesis.workspace_name == workspace_name,
                    models.Hypothesis.observer == observer,
                )
            )
            current_result = await db_session.execute(current_count_stmt)
            current_count = current_result.scalar() or 0

            hypotheses_to_create = count - current_count
            for i in range(hypotheses_to_create):
                await create_test_hypothesis(
                    db_session,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    content=f"Hypothesis {current_count + i + 1}",
                    confidence_score=0.8,
                )

            # Benchmark query
            start_time = time.perf_counter()

            stmt = select(models.Hypothesis).where(
                models.Hypothesis.workspace_name == workspace_name,
                models.Hypothesis.observer == observer,
                models.Hypothesis.observed == observed,
            )
            result = await db_session.execute(stmt)
            hypotheses = list(result.scalars().all())

            end_time = time.perf_counter()
            query_time = (end_time - start_time) * 1000  # Convert to ms

            results.append({
                "record_count": count,
                "query_time_ms": query_time,
                "retrieved": len(hypotheses),
            })

        # Assertions - Query time should scale sub-linearly
        time_100 = results[0]["query_time_ms"]
        time_1000 = results[3]["query_time_ms"]
        scaling_factor = time_1000 / time_100

        assert scaling_factor < 5.0, (
            f"Query scaling factor {scaling_factor:.2f}x is too high "
            f"(10x data should take < 5x time)"
        )

        # Output results
        print("\n📊 Hypothesis Query Scaling Results:")
        print(f"{'Records':<10} {'Query Time (ms)':<18} {'Retrieved':<12}")
        print("-" * 42)
        for r in results:
            print(
                f"{r['record_count']:<10} "
                f"{r['query_time_ms']:<18.2f} "
                f"{r['retrieved']:<12}"
            )
        print(f"\n⚡ Scaling Factor (100→1000): {scaling_factor:.2f}x")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_prediction_query_with_large_hypothesis_set(
        self, db_session: AsyncSession
    ):
        """Test prediction queries with many hypotheses.

        Measures query performance when filtering predictions
        across a large hypothesis set.
        """
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        hypothesis_counts = [10, 25, 50]
        results = []

        for hyp_count in hypothesis_counts:
            # Create hypotheses with predictions
            current_hyp_stmt = (
                select(func.count())
                .select_from(models.Hypothesis)
                .where(
                    models.Hypothesis.workspace_name == workspace_name,
                    models.Hypothesis.observer == observer,
                )
            )
            current_hyp_result = await db_session.execute(current_hyp_stmt)
            current_hyp_count = current_hyp_result.scalar() or 0

            hypotheses_to_create = hyp_count - current_hyp_count
            for i in range(hypotheses_to_create):
                hypothesis = await create_test_hypothesis(
                    db_session,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                    content=f"Hypothesis {current_hyp_count + i + 1}",
                    confidence_score=0.8,
                )

                # Create 5 predictions for each hypothesis
                for j in range(5):
                    await create_test_prediction(
                        db_session,
                        workspace_name=workspace_name,
                        hypothesis_id=hypothesis.id,
                        content=f"Prediction {j+1}",
                        confidence_score=0.7,
                    )

            # Benchmark query
            start_time = time.perf_counter()

            stmt = select(models.Prediction).where(
                models.Prediction.workspace_name == workspace_name,
            )
            result = await db_session.execute(stmt)
            predictions = list(result.scalars().all())

            end_time = time.perf_counter()
            query_time = (end_time - start_time) * 1000  # Convert to ms

            results.append({
                "hypothesis_count": hyp_count,
                "prediction_count": hyp_count * 5,
                "query_time_ms": query_time,
                "retrieved": len(predictions),
            })

        # Output results
        print("\n📊 Prediction Query Scaling Results:")
        print(f"{'Hypotheses':<12} {'Predictions':<12} {'Query Time (ms)':<18}")
        print("-" * 44)
        for r in results:
            print(
                f"{r['hypothesis_count']:<12} "
                f"{r['prediction_count']:<12} "
                f"{r['query_time_ms']:<18.2f}"
            )
