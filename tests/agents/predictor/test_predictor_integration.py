"""Integration tests for the Predictor agent."""

from typing import Any
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.predictor import PredictorAgent, PredictorConfig


class TestPredictorIntegration:
    """Integration test suite for Predictor agent end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_prediction_generation(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test complete prediction generation workflow from hypothesis to storage."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create session
        test_session = models.Session(
            name="test_session",
            workspace_name=workspace.name,
        )
        db_session.add(test_session)
        await db_session.flush()

        # Create collection
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create source premises
        doc1 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User prefers dark mode in the evening",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc1)
        await db_session.flush()

        # Create hypothesis with high confidence
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="User is sensitive to bright light in low-light environments",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.85,
                source_premise_ids=[doc1.id],
                tier=0,
            ),
            workspace.name,
        )

        # Mock LLM call to simulate tool execution
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # Simulate tool calls for predictions
                tool_executor("create_prediction", {
                    "content": "When user opens app after 6 PM, they will enable dark mode within 30 seconds",
                    "specificity": 0.9,
                    "rationale": "Specific time condition and measurable action within timeframe",
                })
                tool_executor("create_prediction", {
                    "content": "User will reduce screen brightness in evening contexts",
                    "specificity": 0.85,
                    "rationale": "Observable action in defined temporal context",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Generated 2 predictions from hypothesis."
            return mock_response

        # Patch honcho_llm_call to use mock
        with patch("src.agents.predictor.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent
            agent = PredictorAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Verify results
        assert result["predictions_created"] == 2
        assert len(result["prediction_ids"]) == 2

        # Verify predictions were stored in database
        pred1 = await crud.prediction.get_prediction(
            db_session, workspace.name, result["prediction_ids"][0]
        )
        assert pred1 is not None
        assert "6 PM" in pred1.content or "evening" in pred1.content.lower()
        assert pred1.hypothesis_id == hypothesis.id
        assert pred1.status == "untested"
        assert pred1.is_blind is True

        pred2 = await crud.prediction.get_prediction(
            db_session, workspace.name, result["prediction_ids"][1]
        )
        assert pred2 is not None
        assert pred2.hypothesis_id == hypothesis.id

    @pytest.mark.asyncio
    async def test_prediction_generation_with_specific_hypothesis(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test prediction generation for a specific hypothesis ID."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create two hypotheses
        hyp1 = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Target hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.8,
                source_premise_ids=["doc1"],
                tier=0,
            ),
            workspace.name,
        )

        hyp2 = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Other hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.9,
                source_premise_ids=["doc2"],
                tier=0,
            ),
            workspace.name,
        )

        # Mock LLM call
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                tool_executor("create_prediction", {
                    "content": "Specific prediction for target hypothesis",
                    "specificity": 0.85,
                    "rationale": "Testing specific hypothesis",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1"]
            mock_response.text = "Generated 1 prediction."
            return mock_response

        with patch("src.agents.predictor.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent with specific hypothesis_id
            agent = PredictorAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
                "hypothesis_id": hyp1.id,
            })

        # Should only generate predictions for target hypothesis
        assert result["predictions_created"] == 1

        pred = await crud.prediction.get_prediction(
            db_session, workspace.name, result["prediction_ids"][0]
        )
        assert pred.hypothesis_id == hyp1.id
        assert pred.hypothesis_id != hyp2.id

    @pytest.mark.asyncio
    async def test_prediction_generation_with_existing_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that agent considers existing predictions."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create hypothesis
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Test hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.8,
                source_premise_ids=["doc1"],
                tier=0,
            ),
            workspace.name,
        )

        # Create existing prediction
        _existing_pred = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Existing prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Mock LLM call - agent should see existing prediction
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            messages = kwargs.get("messages", [])

            # Verify existing prediction is in context
            task_prompt = messages[1]["content"] if len(messages) > 1 else ""
            assert "Existing prediction" in task_prompt

            if tool_executor:
                # Generate a new, different prediction
                tool_executor("create_prediction", {
                    "content": "New prediction different from existing",
                    "specificity": 0.8,
                    "rationale": "Novel prediction",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1"]
            mock_response.text = "Generated new prediction."
            return mock_response

        with patch("src.agents.predictor.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent
            agent = PredictorAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Should create new prediction
        assert result["predictions_created"] == 1

    @pytest.mark.asyncio
    async def test_prediction_generation_with_custom_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test prediction generation respects custom configuration."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create hypothesis
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Test hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.95,
                source_premise_ids=["doc1"],
                tier=0,
            ),
            workspace.name,
        )

        # Custom config with higher specificity threshold
        config = PredictorConfig(
            predictions_per_hypothesis=5,
            specificity_threshold=0.9,
            is_blind=True,
        )

        # Mock LLM to try creating predictions with different specificity levels
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # Try low specificity (should be rejected)
                tool_executor("create_prediction", {
                    "content": "Low specificity prediction",
                    "specificity": 0.7,
                    "rationale": "Too vague",
                })

                # Try high specificity (should be accepted)
                tool_executor("create_prediction", {
                    "content": "High specificity prediction with concrete details",
                    "specificity": 0.95,
                    "rationale": "Very specific and measurable",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Generated predictions."
            return mock_response

        with patch("src.agents.predictor.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent with custom config
            agent = PredictorAgent(db_session, config=config)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Only high-specificity prediction should be created
        assert result["predictions_created"] == 1

        pred = await crud.prediction.get_prediction(
            db_session, workspace.name, result["prediction_ids"][0]
        )
        assert "High specificity" in pred.content
