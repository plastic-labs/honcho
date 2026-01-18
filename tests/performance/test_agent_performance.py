"""Performance benchmarks for individual reasoning agents.

Tests measure execution time for each agent (Abducer, Predictor,
Falsifier, Inductor) in isolation.
"""

import time

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.agents.abducer.agent import AbducerAgent
from src.agents.falsifier.agent import FalsifierAgent
from src.agents.inductor.agent import InductorAgent
from src.agents.predictor.agent import PredictorAgent
from tests.utils.reasoning_test_helpers import (
    create_test_hypothesis,
    create_test_observations,
    create_test_peer,
    create_test_prediction,
    create_test_workspace,
)


class TestAbducerPerformance:
    """Performance benchmarks for Abducer agent."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_abducer_10_observations(
        self, db_session: AsyncSession, mock_abducer_llm
    ):
        """Benchmark Abducer with 10 observations.

        Target: < 5s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        observation_ids = [obs.id for obs in observations]

        # Benchmark
        agent = AbducerAgent(db=db_session)

        start_time = time.perf_counter()

        result = await agent.execute({
            "workspace_name": workspace_name,
            "observer": observer,
            "observed": observed,
        })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert result["hypotheses_created"] > 0
        assert execution_time < 5.0, f"Execution took {execution_time:.2f}s (target: < 5s)"

        print(f"\n⏱️  Abducer time (10 obs): {execution_time:.2f}s")
        print(f"📊 Hypotheses created: {result['hypotheses_created']}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_abducer_50_observations(
        self, db_session: AsyncSession, mock_abducer_llm
    ):
        """Benchmark Abducer with 50 observations.

        Target: < 8s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=50
        )

        observation_ids = [obs.id for obs in observations]

        # Benchmark
        agent = AbducerAgent(db=db_session)

        start_time = time.perf_counter()

        result = await agent.execute({
            "workspace_name": workspace_name,
            "observer": observer,
            "observed": observed,
        })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert result["hypotheses_created"] > 0
        assert execution_time < 8.0, f"Execution took {execution_time:.2f}s (target: < 8s)"

        print(f"\n⏱️  Abducer time (50 obs): {execution_time:.2f}s")
        print(f"📊 Hypotheses created: {result['hypotheses_created']}")


class TestPredictorPerformance:
    """Performance benchmarks for Predictor agent."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_predictor_single_hypothesis(
        self, db_session: AsyncSession, mock_predictor_llm
    ):
        """Benchmark Predictor with single hypothesis.

        Target: < 3s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        # Create observations and hypothesis
        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        # Benchmark
        agent = PredictorAgent(db=db_session)

        start_time = time.perf_counter()

        result = await agent.execute({
            "workspace_name": workspace_name,
            "hypothesis_id": hypothesis.id,
            "observer": observer,
            "observed": observed,
        })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert result["predictions_created"] > 0
        assert execution_time < 3.0, f"Execution took {execution_time:.2f}s (target: < 3s)"

        print(f"\n⏱️  Predictor time (1 hypothesis): {execution_time:.2f}s")
        print(f"📊 Predictions created: {result['predictions_created']}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_predictor_multiple_hypotheses(
        self, db_session: AsyncSession, mock_predictor_llm
    ):
        """Benchmark Predictor with multiple hypotheses.

        Target: < 10s for 5 hypotheses
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        # Create multiple hypotheses
        hypotheses = []
        for i in range(5):
            hyp = await create_test_hypothesis(
                db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=f"Hypothesis {i+1}",
                confidence_score=0.8,
            )
            hypotheses.append(hyp)

        # Benchmark
        agent = PredictorAgent(db=db_session)

        start_time = time.perf_counter()

        total_predictions = 0
        for hypothesis in hypotheses:
            result = await agent.execute({
                "workspace_name": workspace_name,
                "hypothesis_id": hypothesis.id,
                "observer": observer,
                "observed": observed,
            })
            total_predictions += result["predictions_created"]

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert total_predictions > 0
        assert execution_time < 10.0, f"Execution took {execution_time:.2f}s (target: < 10s)"

        print(f"\n⏱️  Predictor time (5 hypotheses): {execution_time:.2f}s")
        print(f"📊 Total predictions: {total_predictions}")
        print(f"📊 Avg time per hypothesis: {execution_time/5:.2f}s")


class TestFalsifierPerformance:
    """Performance benchmarks for Falsifier agent."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_falsifier_single_prediction(
        self, db_session: AsyncSession, mock_falsifier_llm
    ):
        """Benchmark Falsifier with single prediction.

        Target: < 10s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

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
        )

        # Benchmark
        agent = FalsifierAgent(db=db_session)

        start_time = time.perf_counter()

        result = await agent.execute({
            "workspace_name": workspace_name,
            "prediction_id": prediction.id,
            "observer": observer,
            "observed": observed,
        })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert result["prediction_status"] in ["falsified", "unfalsified", "untested"]
        assert execution_time < 10.0, f"Execution took {execution_time:.2f}s (target: < 10s)"

        print(f"\n⏱️  Falsifier time (1 prediction): {execution_time:.2f}s")
        print(f"📊 Status: {result['prediction_status']}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_falsifier_multiple_predictions(
        self, db_session: AsyncSession, mock_falsifier_llm
    ):
        """Benchmark Falsifier with multiple predictions.

        Target: < 30s for 5 predictions
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        # Create multiple predictions
        predictions = []
        for i in range(5):
            pred = await create_test_prediction(
                db_session,
                workspace_name=workspace_name,
                hypothesis_id=hypothesis.id,
                content=f"Prediction {i+1}",
            )
            predictions.append(pred)

        # Benchmark
        agent = FalsifierAgent(db=db_session)

        start_time = time.perf_counter()

        for prediction in predictions:
            await agent.execute({
                "workspace_name": workspace_name,
                "prediction_id": prediction.id,
                "observer": observer,
                "observed": observed,
            })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert execution_time < 30.0, f"Execution took {execution_time:.2f}s (target: < 30s)"

        print(f"\n⏱️  Falsifier time (5 predictions): {execution_time:.2f}s")
        print(f"📊 Avg time per prediction: {execution_time/5:.2f}s")


class TestInductorPerformance:
    """Performance benchmarks for Inductor agent."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_inductor_single_hypothesis(
        self, db_session: AsyncSession, mock_inductor_llm
    ):
        """Benchmark Inductor with single hypothesis.

        Target: < 5s for mocked execution
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=10
        )

        hypothesis = await create_test_hypothesis(
            db_session,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            content="User prefers concise responses",
            confidence_score=0.8,
        )

        # Create multiple unfalsified predictions
        predictions = []
        for i in range(3):
            pred = await create_test_prediction(
                db_session,
                workspace_name=workspace_name,
                hypothesis_id=hypothesis.id,
                content=f"Prediction {i+1}",
                status="unfalsified",
            )
            predictions.append(pred)

        # Benchmark
        agent = InductorAgent(db=db_session)

        start_time = time.perf_counter()

        result = await agent.execute({
            "workspace_name": workspace_name,
            "observer": observer,
            "observed": observed,
        })

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert result["inductions_created"] > 0
        assert execution_time < 5.0, f"Execution took {execution_time:.2f}s (target: < 5s)"

        print(f"\n⏱️  Inductor time (1 hypothesis): {execution_time:.2f}s")
        print(f"📊 Inductions created: {result['inductions_created']}")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_inductor_multiple_hypotheses(
        self, db_session: AsyncSession, mock_inductor_llm
    ):
        """Benchmark Inductor with multiple hypotheses.

        Target: < 15s for 3 hypotheses
        """
        # Setup
        workspace_name = await create_test_workspace(db_session)
        observer = await create_test_peer(db_session, workspace_name, "assistant")
        observed = await create_test_peer(db_session, workspace_name, "user_123")

        observations = await create_test_observations(
            db_session, workspace_name, observer, observed, count=20
        )

        # Create multiple hypotheses with predictions
        total_inductions = 0

        start_time = time.perf_counter()

        agent = InductorAgent(db=db_session)

        for i in range(3):
            hypothesis = await create_test_hypothesis(
                db_session,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                content=f"Hypothesis {i+1}",
                confidence_score=0.8,
            )

            # Create predictions for this hypothesis
            predictions = []
            for j in range(3):
                pred = await create_test_prediction(
                    db_session,
                    workspace_name=workspace_name,
                    hypothesis_id=hypothesis.id,
                    content=f"Prediction {j+1}",
                    status="unfalsified",
                )
                predictions.append(pred)

            result = await agent.execute({
                "workspace_name": workspace_name,
                "observer": observer,
                "observed": observed,
            })
            total_inductions += result["inductions_created"]

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Assertions
        assert total_inductions > 0
        assert execution_time < 15.0, f"Execution took {execution_time:.2f}s (target: < 15s)"

        print(f"\n⏱️  Inductor time (3 hypotheses): {execution_time:.2f}s")
        print(f"📊 Total inductions: {total_inductions}")
        print(f"📊 Avg time per hypothesis: {execution_time/3:.2f}s")
