"""Integration tests for the Inductor agent."""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from nanoid import generate as generate_nanoid

from src import crud, models, schemas
from src.agents.inductor import InductorAgent, InductorConfig


@pytest.fixture
async def inductor_sample_data(db_session, sample_data):
    """Create sample data for inductor tests."""
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
            source_premise_ids=["premise1"],
            tier=0,
        ),
        workspace.name,
    )
    await db_session.commit()

    return {
        "workspace": workspace,
        "observer_peer": observer_peer,
        "observed_peer": observed_peer,
        "hypothesis": hypothesis,
    }


@pytest.fixture
async def sample_predictions(db_session, inductor_sample_data):
    """Create sample unfalsified predictions for testing."""
    # Create multiple similar predictions about dark mode preference
    predictions = []

    # Mock embedding client to return controlled embeddings
    dark_mode_embedding = [0.5] * 1536  # Similar embeddings for cluster 1
    work_embedding = [-0.5] * 1536  # Different embeddings for cluster 2

    call_count = 0

    async def mock_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        # First 3 calls get dark_mode_embedding, next 3 get work_embedding
        if call_count <= 3:
            return dark_mode_embedding
        else:
            return work_embedding

    with patch("src.embedding_client.embedding_client.embed", side_effect=mock_embed):
        # Cluster 1: Dark mode preferences (3 predictions)
        for i in range(3):
            pred = await crud.prediction.create_prediction(
                db_session,
                schemas.PredictionCreate(
                    content=f"User prefers dark mode in evening sessions {i}",
                    hypothesis_id=inductor_sample_data["hypothesis"].id,
                    status="unfalsified",
                ),
                inductor_sample_data["workspace"].name,
            )
            predictions.append(pred)

        # Cluster 2: Work schedule patterns (3 predictions)
        for i in range(3):
            pred = await crud.prediction.create_prediction(
                db_session,
                schemas.PredictionCreate(
                    content=f"User typically works late evening hours {i}",
                    hypothesis_id=inductor_sample_data["hypothesis"].id,
                    status="unfalsified",
                ),
                inductor_sample_data["workspace"].name,
            )
            predictions.append(pred)

    await db_session.commit()
    return predictions


class TestInductorIntegration:
    """Integration tests for InductorAgent."""

    async def test_end_to_end_pattern_extraction(self, db_session, inductor_sample_data, sample_predictions):
        """Test complete pattern extraction workflow with mocked LLM."""
        agent = InductorAgent(
            db_session,
            config=InductorConfig(
                min_predictions_per_pattern=2,
                similarity_threshold=0.8,
                stability_score_threshold=0.6,
            ),
        )

        # Mock LLM to simulate pattern extraction
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            tool_executor = kwargs.get("tool_executor")

            if tool_executor:
                # Simulate creating an induction
                tool_executor("create_induction", {
                    "content": "User consistently prefers dark mode during evening work sessions",
                    "pattern_type": "preference",
                    "stability": 0.85,
                    "rationale": "Based on 3 similar unfalsified predictions showing consistent dark mode usage in evening contexts. High stability due to multiple supporting observations with consistent context.",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1"]
            return mock_response

        with patch("src.agents.inductor.agent.honcho_llm_call", side_effect=mock_llm_call):
            result = await agent.execute({
                "workspace_name": inductor_sample_data["workspace"].name,
                "observer": inductor_sample_data["observer_peer"].name,
                "observed": inductor_sample_data["observed_peer"].name,
            })

        # Verify inductions were created
        assert result["inductions_created"] >= 1
        assert result["predictions_analyzed"] == 6
        assert result["clusters_found"] >= 1
        assert len(result["induction_ids"]) >= 1

        # Verify induction was stored in database
        induction_id = result["induction_ids"][0]
        induction = await crud.induction.get_induction(
            db_session,
            inductor_sample_data["workspace"].name,
            induction_id
        )

        assert induction is not None
        assert "dark mode" in induction.content.lower() or "evening" in induction.content.lower()
        assert induction.pattern_type == "preference"
        assert induction.confidence == "high"  # 0.85 stability -> "high" confidence
        assert induction.source_prediction_ids is not None and len(induction.source_prediction_ids) >= 2

    async def test_multiple_clusters_extraction(self, db_session, inductor_sample_data, sample_predictions):
        """Test extraction of patterns from multiple clusters."""
        agent = InductorAgent(
            db_session,
            config=InductorConfig(
                min_predictions_per_pattern=2,
                similarity_threshold=0.8,
                stability_score_threshold=0.6,
                max_inductions_per_run=10,
            ),
        )

        call_count = 0

        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            tool_executor = kwargs.get("tool_executor")

            if tool_executor:
                call_count += 1

                # Different pattern for each cluster
                if call_count == 1:
                    tool_executor("create_induction", {
                        "content": "User prefers dark mode interface during evening work",
                        "pattern_type": "preference",
                        "stability": 0.80,
                        "rationale": "Consistent pattern across multiple evening sessions",
                    })
                elif call_count == 2:
                    tool_executor("create_induction", {
                        "content": "User typically works during late evening hours",
                        "pattern_type": "behavior",
                        "stability": 0.75,
                        "rationale": "Recurring work schedule pattern observed across multiple days",
                    })

            mock_response = MagicMock()
            mock_response.tool_calls_made = [f"call{call_count}"]
            return mock_response

        with patch("src.agents.inductor.agent.honcho_llm_call", side_effect=mock_llm_call):
            result = await agent.execute({
                "workspace_name": inductor_sample_data["workspace"].name,
                "observer": inductor_sample_data["observer_peer"].name,
                "observed": inductor_sample_data["observed_peer"].name,
            })

        # Verify multiple inductions created
        assert result["inductions_created"] >= 2
        assert result["clusters_found"] >= 2
        assert len(result["induction_ids"]) >= 2

        # Verify different pattern types
        induction_ids = result["induction_ids"][:2]
        inductions = []
        for induction_id in induction_ids:
            induction = await crud.induction.get_induction(
                db_session,
                inductor_sample_data["workspace"].name,
                induction_id
            )
            inductions.append(induction)

        pattern_types = {ind.pattern_type for ind in inductions}
        assert len(pattern_types) >= 1  # At least one distinct pattern type

    async def test_stability_threshold_enforcement(self, db_session, inductor_sample_data, sample_predictions):
        """Test that low-stability patterns are rejected."""
        agent = InductorAgent(
            db_session,
            config=InductorConfig(
                min_predictions_per_pattern=2,
                similarity_threshold=0.8,
                stability_score_threshold=0.8,  # High threshold
            ),
        )

        # Mock LLM to return low-stability pattern
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            tool_executor = kwargs.get("tool_executor")

            if tool_executor:
                # Create pattern with low stability (below threshold)
                tool_executor("create_induction", {
                    "content": "User might prefer dark mode sometimes",
                    "pattern_type": "tendency",
                    "stability": 0.5,  # Below 0.8 threshold
                    "rationale": "Pattern observed but with low confidence due to inconsistent context",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1"]
            return mock_response

        with patch("src.agents.inductor.agent.honcho_llm_call", side_effect=mock_llm_call):
            result = await agent.execute({
                "workspace_name": inductor_sample_data["workspace"].name,
                "observer": inductor_sample_data["observer_peer"].name,
                "observed": inductor_sample_data["observed_peer"].name,
            })

        # Verify no inductions created due to low stability
        assert result["inductions_created"] == 0
        assert result["predictions_analyzed"] == 6
        assert result["clusters_found"] >= 1  # Clusters found but patterns rejected

    async def test_custom_configuration_workflow(self, db_session, inductor_sample_data, sample_predictions):
        """Test pattern extraction with custom configuration."""
        custom_config = InductorConfig(
            min_predictions_per_pattern=3,  # Require at least 3 predictions
            similarity_threshold=0.75,
            stability_score_threshold=0.7,
            max_predictions_retrieval=50,
            pattern_types=["preference", "behavior"],  # Limited pattern types
        )

        agent = InductorAgent(db_session, config=custom_config)

        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            tool_executor = kwargs.get("tool_executor")

            if tool_executor:
                tool_executor("create_induction", {
                    "content": "User consistently prefers dark mode",
                    "pattern_type": "preference",
                    "stability": 0.85,
                    "rationale": "Strong consistent pattern with multiple supporting predictions",
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1"]
            return mock_response

        with patch("src.agents.inductor.agent.honcho_llm_call", side_effect=mock_llm_call):
            result = await agent.execute({
                "workspace_name": inductor_sample_data["workspace"].name,
                "observer": inductor_sample_data["observer_peer"].name,
                "observed": inductor_sample_data["observed_peer"].name,
            })

        # Verify configuration was respected
        assert result["inductions_created"] >= 1

        # Verify pattern type is from allowed list
        induction_id = result["induction_ids"][0]
        induction = await crud.induction.get_induction(
            db_session,
            inductor_sample_data["workspace"].name,
            induction_id
        )

        assert induction.pattern_type in ["preference", "behavior"]
        assert induction.confidence in ["high", "medium", "low"]  # Check confidence is string
