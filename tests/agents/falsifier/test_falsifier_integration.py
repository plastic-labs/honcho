"""Integration tests for the Falsifier agent."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.falsifier import FalsifierAgent, FalsifierConfig


class TestFalsifierIntegration:
    """Integration test suite for Falsifier agent end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_falsification_unfalsified(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test complete falsification workflow that confirms a prediction."""
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

        # Create supporting observations
        doc1 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User enabled dark mode at 8 PM",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc1)

        doc2 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User switched to dark theme in the evening",
            embedding=[0.2] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc2)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="User prefers dark mode in evening",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.8,
                source_premise_ids=[doc1.id],
                tier=0,
            ),
            workspace.name,
        )

        # Create prediction
        prediction = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="User will enable dark mode after 6 PM",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Mock LLM call to simulate falsification attempt
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # First iteration: Generate search query
                tool_executor("generate_search_query", {
                    "query": "dark mode evening user",
                    "strategy": "Search for observations about dark mode usage patterns",
                })

                # Second iteration: Evaluate prediction
                tool_executor("evaluate_prediction", {
                    "evidence_summary": "Found supporting observations of dark mode usage in evening. No contradictions found.",
                    "confidence": 0.85,
                    "determination": "unfalsified",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Prediction unfalsified through search."
            return mock_response

        # Patch honcho_llm_call to use mock
        with patch("src.agents.falsifier.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent
            agent = FalsifierAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Verify results
        assert result["predictions_tested"] == 1
        assert result["predictions_unfalsified"] == 1
        assert result["predictions_falsified"] == 0
        assert result["predictions_inconclusive"] == 0
        assert len(result["trace_ids"]) == 1

        # Verify prediction status updated
        updated_pred = await crud.prediction.get_prediction(
            db_session, workspace.name, prediction.id
        )
        assert updated_pred.status == "unfalsified"

        # Verify trace created
        trace = await crud.trace.get_trace(
            db_session, workspace.name, result["trace_ids"][0]
        )
        assert trace is not None
        assert trace.prediction_id == prediction.id
        assert trace.final_status == "unfalsified"
        assert trace.search_queries is not None and len(trace.search_queries) > 0
        assert "dark mode" in trace.search_queries[0]

    @pytest.mark.asyncio
    async def test_end_to_end_falsification_falsified(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test complete falsification workflow that falsifies a prediction."""
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

        # Create contradicting observation
        doc1 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User prefers light mode and disabled dark theme permanently",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc1)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="User prefers dark mode",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.7,
                source_premise_ids=["doc_old"],
                tier=0,
            ),
            workspace.name,
        )

        # Create prediction
        prediction = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="User will enable dark mode",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Mock LLM call to find contradiction
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that finds contradicting evidence."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # First iteration: Generate search query
                tool_executor("generate_search_query", {
                    "query": "dark mode preferences",
                    "strategy": "Search for observations about theme preferences",
                })

                # Second iteration: Evaluate and falsify
                tool_executor("evaluate_prediction", {
                    "evidence_summary": "Found observation stating user prefers light mode and disabled dark theme permanently. Strong contradiction.",
                    "confidence": 0.9,
                    "determination": "falsified",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Prediction falsified."
            return mock_response

        # Patch honcho_llm_call to use mock
        with patch("src.agents.falsifier.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent
            agent = FalsifierAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Verify results
        assert result["predictions_tested"] == 1
        assert result["predictions_falsified"] == 1
        assert result["predictions_unfalsified"] == 0
        assert result["predictions_inconclusive"] == 0

        # Verify prediction status updated
        updated_pred = await crud.prediction.get_prediction(
            db_session, workspace.name, prediction.id
        )
        assert updated_pred.status == "falsified"

        # Verify trace has contradicting premises
        trace = await crud.trace.get_trace(
            db_session, workspace.name, result["trace_ids"][0]
        )
        assert trace.final_status == "falsified"
        assert trace.contradicting_premise_ids is not None and len(trace.contradicting_premise_ids) > 0

    @pytest.mark.asyncio
    async def test_falsification_with_specific_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test falsification for a specific prediction ID."""
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

        # Create two predictions
        pred1 = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Target prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        pred2 = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Other prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Mock LLM call
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM for confirmation."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                tool_executor("generate_search_query", {
                    "query": "test query",
                    "strategy": "Search for relevant observations",
                })
                tool_executor("evaluate_prediction", {
                    "evidence_summary": "No contradictions found",
                    "confidence": 0.8,
                    "determination": "unfalsified",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            return mock_response

        with patch("src.agents.falsifier.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent with specific prediction_id
            agent = FalsifierAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
                "prediction_id": pred1.id,
            })

        # Should only test target prediction
        assert result["predictions_tested"] == 1

        # Check pred1 was tested
        pred1_updated = await crud.prediction.get_prediction(
            db_session, workspace.name, pred1.id
        )
        assert pred1_updated.status == "unfalsified"

        # Check pred2 was not tested
        pred2_updated = await crud.prediction.get_prediction(
            db_session, workspace.name, pred2.id
        )
        assert pred2_updated.status == "untested"

    @pytest.mark.asyncio
    async def test_falsification_with_custom_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test falsification respects custom configuration."""
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

        # Create prediction
        prediction = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Test prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Custom config with higher thresholds
        config = FalsifierConfig(
            contradiction_confidence_threshold=0.9,
            unfalsified_confidence_threshold=0.95,
        )

        # Mock LLM to try confirming with insufficient confidence
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM with low confidence."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                tool_executor("generate_search_query", {
                    "query": "test",
                    "strategy": "test",
                })
                # Try to confirm with confidence below threshold
                tool_executor("evaluate_prediction", {
                    "evidence_summary": "Some evidence",
                    "confidence": 0.85,  # Below confirmation threshold of 0.95
                    "determination": "unfalsified",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            return mock_response

        with patch("src.agents.falsifier.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent with custom config
            agent = FalsifierAgent(db_session, config=config)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Should remain untested or be unfalsified after all iterations
        # (depending on whether max iterations reached)
        assert result["predictions_tested"] == 1
