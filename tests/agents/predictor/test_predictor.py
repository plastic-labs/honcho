"""Unit tests for the Predictor agent."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.predictor import PredictorAgent, PredictorConfig


class TestPredictorAgent:
    """Test suite for Predictor agent."""

    @pytest.mark.asyncio
    async def test_validate_input_success(
        self,
        db_session: AsyncSession,
    ):
        """Test that valid input passes validation."""
        agent = PredictorAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "test_observer",
            "observed": "test_observed",
        }

        assert agent.validate_input(input_data) is True

    @pytest.mark.asyncio
    async def test_validate_input_with_hypothesis_id(
        self,
        db_session: AsyncSession,
    ):
        """Test that valid input with hypothesis_id passes validation."""
        agent = PredictorAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "test_observer",
            "observed": "test_observed",
            "hypothesis_id": "hyp123",
        }

        assert agent.validate_input(input_data) is True

    @pytest.mark.asyncio
    async def test_validate_input_missing_field(
        self,
        db_session: AsyncSession,
    ):
        """Test that missing required fields raise ValueError."""
        agent = PredictorAgent(db_session)

        with pytest.raises(ValueError, match="Missing required field"):
            agent.validate_input({"workspace_name": "test"})

    @pytest.mark.asyncio
    async def test_validate_input_wrong_type(
        self,
        db_session: AsyncSession,
    ):
        """Test that wrong field types raise ValueError."""
        agent = PredictorAgent(db_session)

        with pytest.raises(ValueError, match="must be a string"):
            agent.validate_input({
                "workspace_name": 123,
                "observer": "test",
                "observed": "test",
            })

    @pytest.mark.asyncio
    async def test_validate_input_wrong_hypothesis_id_type(
        self,
        db_session: AsyncSession,
    ):
        """Test that wrong hypothesis_id type raises ValueError."""
        agent = PredictorAgent(db_session)

        with pytest.raises(ValueError, match="hypothesis_id must be a string"):
            agent.validate_input({
                "workspace_name": "test",
                "observer": "test",
                "observed": "test",
                "hypothesis_id": 123,
            })

    @pytest.mark.asyncio
    async def test_execute_no_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that no hypotheses returns early with reason."""
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

        # Execute agent (no hypotheses exist)
        agent = PredictorAgent(db_session)
        result = await agent.run({
            "workspace_name": workspace.name,
            "observer": observer_peer.name,
            "observed": observed_peer.name,
        })

        # Should return with no_hypotheses reason
        assert result["predictions_created"] == 0
        assert result["prediction_ids"] == []
        assert result["reason"] == "no_hypotheses"

    @pytest.mark.asyncio
    async def test_retrieve_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test hypothesis retrieval filters by confidence and status."""
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

        # Create high-confidence hypothesis (should be included)
        high_conf = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="High confidence hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.8,
                source_premise_ids=["doc1"],
                tier=0,
            ),
            workspace.name,
        )

        # Create low-confidence hypothesis (should be excluded)
        await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Low confidence hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.3,
                source_premise_ids=["doc2"],
                tier=0,
            ),
            workspace.name,
        )

        # Create falsified hypothesis (should be excluded)
        await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Falsified hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="falsified",
                confidence=0.9,
                source_premise_ids=["doc3"],
                tier=0,
            ),
            workspace.name,
        )

        # Test retrieval
        agent = PredictorAgent(db_session)
        hypotheses = await agent._retrieve_hypotheses(
            workspace.name, observer_peer.name, observed_peer.name
        )

        # Should only return high-confidence active hypothesis
        assert len(hypotheses) == 1
        assert hypotheses[0].id == high_conf.id
        assert hypotheses[0].confidence >= 0.6

    @pytest.mark.asyncio
    async def test_retrieve_existing_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test existing prediction retrieval."""
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

        # Create prediction for this hypothesis
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

        # Create prediction for different hypothesis
        other_hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Other hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.7,
                source_premise_ids=["doc2"],
                tier=0,
            ),
            workspace.name,
        )

        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="Other prediction",
                hypothesis_id=other_hypothesis.id,
                status="untested",
                is_blind=True,
            ),
            workspace.name,
        )

        # Test retrieval
        agent = PredictorAgent(db_session)
        predictions = await agent._retrieve_existing_predictions(
            workspace.name, hypothesis.id
        )

        # Should only return predictions for the specific hypothesis
        assert len(predictions) == 1
        assert predictions[0].id == prediction.id
        assert predictions[0].hypothesis_id == hypothesis.id

    @pytest.mark.asyncio
    async def test_retrieve_source_premises(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test source premise retrieval."""
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

        # Create documents
        doc1 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="First premise",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc1)

        doc2 = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="Second premise",
            embedding=[0.2] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc2)
        await db_session.flush()

        # Test retrieval
        agent = PredictorAgent(db_session)
        premises = await agent._retrieve_source_premises(
            workspace.name, [doc1.id, doc2.id]
        )

        # Should return both documents
        assert len(premises) == 2
        assert {p.id for p in premises} == {doc1.id, doc2.id}

    @pytest.mark.asyncio
    async def test_retrieve_source_premises_empty(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test source premise retrieval with empty list."""
        workspace, _observer_peer = sample_data

        # Test retrieval with empty list
        agent = PredictorAgent(db_session)
        premises = await agent._retrieve_source_premises(workspace.name, [])

        # Should return empty list
        assert len(premises) == 0

    @pytest.mark.asyncio
    async def test_store_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test storing predictions in database."""
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

        # Prepare prediction data
        predictions_data = [
            {
                "content": "When user opens app after 6 PM, they will enable dark mode",
                "specificity": 0.85,
                "rationale": "Tests preference for dark mode in evening",
            },
            {
                "content": "User will choose vegetarian option when ordering from restaurants",
                "specificity": 0.80,
                "rationale": "Tests dietary preferences",
            },
        ]

        # Store predictions
        agent = PredictorAgent(db_session)
        prediction_ids = await agent._store_predictions(
            predictions_data, hypothesis.id, workspace.name
        )

        # Verify storage
        assert len(prediction_ids) == 2

        # Retrieve and verify first prediction
        pred1 = await crud.prediction.get_prediction(
            db_session, workspace.name, prediction_ids[0]
        )
        assert pred1.content == "When user opens app after 6 PM, they will enable dark mode"
        assert pred1.hypothesis_id == hypothesis.id
        assert pred1.status == "untested"
        assert pred1.is_blind is True

    @pytest.mark.asyncio
    async def test_custom_config(
        self,
        db_session: AsyncSession,
    ):
        """Test agent with custom configuration."""
        config = PredictorConfig(
            predictions_per_hypothesis=5,
            min_confidence_threshold=0.7,
            specificity_threshold=0.8,
            max_hypothesis_retrieval=10,
            novelty_threshold=0.9,
            is_blind=True,
        )

        agent = PredictorAgent(db_session, config=config)

        assert agent.config.predictions_per_hypothesis == 5
        assert agent.config.min_confidence_threshold == 0.7
        assert agent.config.specificity_threshold == 0.8
        assert agent.config.max_hypothesis_retrieval == 10
        assert agent.config.novelty_threshold == 0.9
        assert agent.config.is_blind is True

    @pytest.mark.asyncio
    async def test_get_predictor_tools(
        self,
        db_session: AsyncSession,
    ):
        """Test tool definitions for predictor."""
        agent = PredictorAgent(db_session)
        tools = agent._get_predictor_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "create_prediction"
        assert "input_schema" in tools[0]

        # Verify required fields
        properties = tools[0]["input_schema"]["properties"]
        assert "content" in properties
        assert "specificity" in properties
        assert "rationale" in properties

        required = tools[0]["input_schema"]["required"]
        assert "content" in required
        assert "specificity" in required
        assert "rationale" in required
