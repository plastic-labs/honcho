"""Unit tests for the Falsifier agent."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.falsifier import FalsifierAgent, FalsifierConfig


class TestFalsifierAgent:
    """Test suite for Falsifier agent."""

    @pytest.mark.asyncio
    async def test_validate_input_success(
        self,
        db_session: AsyncSession,
    ):
        """Test that valid input passes validation."""
        agent = FalsifierAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "test_observer",
            "observed": "test_observed",
        }

        assert agent.validate_input(input_data) is True

    @pytest.mark.asyncio
    async def test_validate_input_with_prediction_id(
        self,
        db_session: AsyncSession,
    ):
        """Test that valid input with prediction_id passes validation."""
        agent = FalsifierAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "test_observer",
            "observed": "test_observed",
            "prediction_id": "pred123",
        }

        assert agent.validate_input(input_data) is True

    @pytest.mark.asyncio
    async def test_validate_input_missing_field(
        self,
        db_session: AsyncSession,
    ):
        """Test that missing required fields raise ValueError."""
        agent = FalsifierAgent(db_session)

        with pytest.raises(ValueError, match="Missing required field"):
            agent.validate_input({"workspace_name": "test"})

    @pytest.mark.asyncio
    async def test_validate_input_wrong_type(
        self,
        db_session: AsyncSession,
    ):
        """Test that wrong field types raise ValueError."""
        agent = FalsifierAgent(db_session)

        with pytest.raises(ValueError, match="must be a string"):
            agent.validate_input({
                "workspace_name": 123,
                "observer": "test",
                "observed": "test",
            })

    @pytest.mark.asyncio
    async def test_validate_input_wrong_prediction_id_type(
        self,
        db_session: AsyncSession,
    ):
        """Test that wrong prediction_id type raises ValueError."""
        agent = FalsifierAgent(db_session)

        with pytest.raises(ValueError, match="prediction_id must be a string"):
            agent.validate_input({
                "workspace_name": "test",
                "observer": "test",
                "observed": "test",
                "prediction_id": 123,
            })

    @pytest.mark.asyncio
    async def test_execute_no_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that no predictions returns early with reason."""
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

        # Execute agent (no predictions exist)
        agent = FalsifierAgent(db_session)
        result = await agent.run({
            "workspace_name": workspace.name,
            "observer": observer_peer.name,
            "observed": observed_peer.name,
        })

        # Should return with no_predictions reason
        assert result["predictions_tested"] == 0
        assert result["predictions_falsified"] == 0
        assert result["predictions_unfalsified"] == 0
        assert result["predictions_inconclusive"] == 0
        assert result["trace_ids"] == []
        assert result["reason"] == "no_predictions"

    @pytest.mark.asyncio
    async def test_retrieve_predictions_filters_untested(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test prediction retrieval filters by untested status."""
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

        # Create untested prediction (should be included)
        untested_pred = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Untested prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Create falsified prediction (should be excluded)
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Falsified prediction",
                hypothesis_id=hypothesis.id,
                status="falsified",
                is_blind=True,
            ),
            workspace.name,
        )

        # Create unfalsified prediction (should be excluded)
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Confirmed prediction",
                hypothesis_id=hypothesis.id,
                status="unfalsified",
                is_blind=True,
            ),
            workspace.name,
        )

        # Test retrieval
        agent = FalsifierAgent(db_session)
        predictions = await agent._retrieve_predictions(
            workspace.name, observer_peer.name, observed_peer.name
        )

        # Should only return untested prediction
        assert len(predictions) == 1
        assert predictions[0].id == untested_pred.id
        assert predictions[0].status == "untested"

    @pytest.mark.asyncio
    async def test_retrieve_specific_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving specific prediction by ID."""
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

        # Create specific prediction
        target_pred = await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Target prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Create another prediction
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Other prediction",
                hypothesis_id=hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Test retrieval with specific ID
        agent = FalsifierAgent(db_session)
        predictions = await agent._retrieve_predictions(
            workspace.name, observer_peer.name, observed_peer.name, target_pred.id
        )

        # Should only return target prediction
        assert len(predictions) == 1
        assert predictions[0].id == target_pred.id

    @pytest.mark.asyncio
    async def test_search_observations(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test searching observations for contradictions."""
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

        # Create observations
        doc1 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User prefers light mode in the morning",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc1)
        await db_session.flush()

        # Test search
        agent = FalsifierAgent(db_session)
        results = await agent._search_observations(
            workspace.name,
            observer_peer.name,
            observed_peer.name,
            "user mode preferences",
        )

        # Should find the observation
        assert len(results) > 0
        assert doc1.id in [doc.id for doc in results]

    @pytest.mark.asyncio
    async def test_custom_config(
        self,
        db_session: AsyncSession,
    ):
        """Test agent with custom configuration."""
        config = FalsifierConfig(
            max_search_iterations=10,
            contradiction_confidence_threshold=0.8,
            unfalsified_confidence_threshold=0.9,
            search_efficiency_target=0.7,
            max_predictions_per_run=5,
            search_result_limit=10,
        )

        agent = FalsifierAgent(db_session, config=config)

        assert agent.config.max_search_iterations == 10
        assert agent.config.contradiction_confidence_threshold == 0.8
        assert agent.config.unfalsified_confidence_threshold == 0.9
        assert agent.config.search_efficiency_target == 0.7
        assert agent.config.max_predictions_per_run == 5
        assert agent.config.search_result_limit == 10

    @pytest.mark.asyncio
    async def test_get_falsifier_tools(
        self,
        db_session: AsyncSession,
    ):
        """Test tool definitions for falsifier."""
        agent = FalsifierAgent(db_session)
        tools = agent._get_falsifier_tools()

        assert len(tools) == 3

        # Check tool names
        tool_names = {tool["name"] for tool in tools}
        assert tool_names == {"generate_search_query", "evaluate_prediction", "stop_search"}

        # Verify generate_search_query tool
        search_tool = next(t for t in tools if t["name"] == "generate_search_query")
        assert "input_schema" in search_tool
        properties = search_tool["input_schema"]["properties"]
        assert "query" in properties
        assert "strategy" in properties
        required = search_tool["input_schema"]["required"]
        assert "query" in required
        assert "strategy" in required

        # Verify evaluate_prediction tool
        eval_tool = next(t for t in tools if t["name"] == "evaluate_prediction")
        properties = eval_tool["input_schema"]["properties"]
        assert "evidence_summary" in properties
        assert "confidence" in properties
        assert "determination" in properties
        required = eval_tool["input_schema"]["required"]
        assert "evidence_summary" in required
        assert "confidence" in required
        assert "determination" in required

        # Verify stop_search tool
        stop_tool = next(t for t in tools if t["name"] == "stop_search")
        properties = stop_tool["input_schema"]["properties"]
        assert "reason" in properties
        required = stop_tool["input_schema"]["required"]
        assert "reason" in required
