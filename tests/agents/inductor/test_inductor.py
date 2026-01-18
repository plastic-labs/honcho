"""Unit tests for the Inductor agent."""

import pytest
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


class TestInductorAgent:
    """Test suite for InductorAgent."""

    async def test_validate_input_success(self, db_session):
        """Test successful input validation."""
        agent = InductorAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "agent_peer",
            "observed": "user_peer",
        }

        assert agent.validate_input(input_data) is True

    async def test_validate_input_missing_field(self, db_session):
        """Test validation fails with missing required field."""
        agent = InductorAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "agent_peer",
            # Missing 'observed'
        }

        with pytest.raises(ValueError, match="Missing required field: observed"):
            agent.validate_input(input_data)

    async def test_validate_input_wrong_type(self, db_session):
        """Test validation fails with wrong field type."""
        agent = InductorAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": 123,  # Should be string
            "observed": "user_peer",
        }

        with pytest.raises(ValueError, match="observer must be a string"):
            agent.validate_input(input_data)

    async def test_execute_insufficient_predictions(self, db_session, inductor_sample_data):
        """Test execute returns early when insufficient predictions."""
        agent = InductorAgent(db_session, config=InductorConfig(min_predictions_per_pattern=5))

        # Create only 2 predictions (less than minimum)
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Prediction 1",
                hypothesis_id=inductor_sample_data["hypothesis"].id,
                status="unfalsified",
            ),
            inductor_sample_data["workspace"].name,
        )
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Prediction 2",
                hypothesis_id=inductor_sample_data["hypothesis"].id,
                status="unfalsified",
            ),
            inductor_sample_data["workspace"].name,
        )
        await db_session.commit()

        result = await agent.execute({
            "workspace_name": inductor_sample_data["workspace"].name,
            "observer": inductor_sample_data["observer_peer"].name,
            "observed": inductor_sample_data["observed_peer"].name,
        })

        assert result["inductions_created"] == 0
        assert result["reason"] == "insufficient_predictions"
        assert result["predictions_analyzed"] == 2

    async def test_execute_no_clusters(self, db_session, inductor_sample_data):
        """Test execute returns early when no clusters found."""
        agent = InductorAgent(
            db_session,
            config=InductorConfig(
                min_predictions_per_pattern=3,
                similarity_threshold=0.99,  # Very high threshold
            ),
        )

        # Create 3 dissimilar predictions
        for i in range(3):
            embedding = [0.0] * 1536
            embedding[i * 100] = 1.0  # Make them very different

            await crud.prediction.create_prediction(
                db_session,
                schemas.PredictionCreate(
                    content=f"Very different prediction {i}",
                    hypothesis_id=inductor_sample_data["hypothesis"].id,
                    status="unfalsified",
                ),
                inductor_sample_data["workspace"].name,
            )
        await db_session.commit()

        result = await agent.execute({
            "workspace_name": inductor_sample_data["workspace"].name,
            "observer": inductor_sample_data["observer_peer"].name,
            "observed": inductor_sample_data["observed_peer"].name,
        })

        assert result["inductions_created"] == 0
        assert result["reason"] == "no_clusters"
        assert result["clusters_found"] == 0

    async def test_cosine_similarity(self, db_session):
        """Test cosine similarity calculation."""
        agent = InductorAgent(db_session)

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert agent._cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert agent._cosine_similarity(vec1, vec2) == pytest.approx(0.0)

        # Opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert agent._cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

        # Zero vector
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert agent._cosine_similarity(vec1, vec2) == 0.0

    async def test_clustering_logic(self, db_session, inductor_sample_data):
        """Test prediction clustering by similarity."""
        agent = InductorAgent(
            db_session,
            config=InductorConfig(
                min_predictions_per_pattern=2,
                similarity_threshold=0.9,  # High threshold for this test
            ),
        )

        # Mock embedding client to return controlled embeddings
        base_embedding1 = [1.0] + [0.0] * 1535  # Unit vector in first dimension
        base_embedding2 = [0.0] * 1535 + [1.0]  # Unit vector in last dimension (orthogonal)

        call_count = 0

        async def mock_embed(text: str) -> list[float]:
            nonlocal call_count
            # First 3 calls get embedding1, next 2 get embedding2
            call_count += 1
            if call_count <= 3:
                return base_embedding1
            else:
                return base_embedding2

        # Create 3 predictions with similar embeddings (cluster 1)
        predictions_cluster1 = []
        with patch("src.embedding_client.embedding_client.embed", side_effect=mock_embed):
            for i in range(3):
                pred = await crud.prediction.create_prediction(
                    db_session,
                    schemas.PredictionCreate(
                        content=f"User prefers dark mode {i}",
                        hypothesis_id=inductor_sample_data["hypothesis"].id,
                        status="unfalsified",
                    ),
                    inductor_sample_data["workspace"].name,
                )
                predictions_cluster1.append(pred)

            # Create 2 predictions with different embeddings (cluster 2)
            predictions_cluster2 = []
            for i in range(2):
                pred = await crud.prediction.create_prediction(
                    db_session,
                    schemas.PredictionCreate(
                        content=f"User works in evening {i}",
                        hypothesis_id=inductor_sample_data["hypothesis"].id,
                        status="unfalsified",
                    ),
                    inductor_sample_data["workspace"].name,
                )
                predictions_cluster2.append(pred)

        await db_session.commit()

        # Get all predictions
        all_predictions = predictions_cluster1 + predictions_cluster2

        # Cluster them
        clusters = await agent._cluster_predictions(all_predictions)

        # Should find 2 clusters (identical embeddings within each cluster)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"
        # Check cluster sizes
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        assert cluster_sizes == [3, 2], f"Expected cluster sizes [3, 2], got {cluster_sizes}"

    async def test_retrieve_hypotheses(self, db_session, inductor_sample_data):
        """Test hypothesis retrieval by IDs."""
        agent = InductorAgent(db_session)

        # Create additional hypothesis
        hypothesis2 = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Second hypothesis",
                observer=inductor_sample_data["observer_peer"].name,
                observed=inductor_sample_data["observed_peer"].name,
                status="active",
                confidence=0.8,
                source_premise_ids=["premise2"],
                tier=0,
            ),
            inductor_sample_data["workspace"].name,
        )
        await db_session.commit()

        # Retrieve both
        hypotheses = await agent._retrieve_hypotheses(
            inductor_sample_data["workspace"].name,
            [inductor_sample_data["hypothesis"].id, hypothesis2.id],
        )

        assert len(hypotheses) == 2
        assert {h.id for h in hypotheses} == {inductor_sample_data["hypothesis"].id, hypothesis2.id}

    async def test_custom_config(self, db_session):
        """Test agent with custom configuration."""
        custom_config = InductorConfig(
            min_predictions_per_pattern=5,
            similarity_threshold=0.85,
            stability_score_threshold=0.7,
            max_predictions_retrieval=50,
            max_inductions_per_run=5,
        )

        agent = InductorAgent(db_session, config=custom_config)

        assert agent.config.min_predictions_per_pattern == 5
        assert agent.config.similarity_threshold == 0.85
        assert agent.config.stability_score_threshold == 0.7
        assert agent.config.max_predictions_retrieval == 50
        assert agent.config.max_inductions_per_run == 5

    async def test_get_inductor_tools(self, db_session):
        """Test inductor tool definitions."""
        agent = InductorAgent(db_session)

        tools = agent._get_inductor_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "create_induction"

        # Verify schema structure
        schema = tools[0]["input_schema"]
        assert schema["type"] == "object"
        assert set(schema["properties"].keys()) == {"content", "pattern_type", "stability", "rationale"}
        assert schema["required"] == ["content", "pattern_type", "stability", "rationale"]

        # Verify pattern types in enum
        pattern_type_enum = schema["properties"]["pattern_type"]["enum"]
        assert "preference" in pattern_type_enum
        assert "behavior" in pattern_type_enum
        assert "personality" in pattern_type_enum
