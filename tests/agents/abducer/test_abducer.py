"""Unit tests for the Abducer agent."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.abducer import AbducerAgent, AbducerConfig


class TestAbducerAgent:
    """Test suite for Abducer agent."""

    @pytest.mark.asyncio
    async def test_validate_input_success(
        self,
        db_session: AsyncSession,
    ):
        """Test that valid input passes validation."""
        agent = AbducerAgent(db_session)

        input_data = {
            "workspace_name": "test_workspace",
            "observer": "test_observer",
            "observed": "test_observed",
        }

        assert agent.validate_input(input_data) is True

    @pytest.mark.asyncio
    async def test_validate_input_missing_field(
        self,
        db_session: AsyncSession,
    ):
        """Test that missing required fields raise ValueError."""
        agent = AbducerAgent(db_session)

        with pytest.raises(ValueError, match="Missing required field"):
            agent.validate_input({"workspace_name": "test"})

    @pytest.mark.asyncio
    async def test_validate_input_wrong_type(
        self,
        db_session: AsyncSession,
    ):
        """Test that wrong field types raise ValueError."""
        agent = AbducerAgent(db_session)

        with pytest.raises(ValueError, match="must be a string"):
            agent.validate_input({
                "workspace_name": 123,
                "observer": "test",
                "observed": "test",
            })

    @pytest.mark.asyncio
    async def test_execute_insufficient_premises(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that insufficient premises returns early with reason."""
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

        # Create only 1 premise (below min_premise_count=3)
        doc = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="Test observation",
            embedding=[0.1] * 1536,
            session_name="test_session",
            level="explicit",
        )
        db_session.add(doc)
        await db_session.flush()

        # Execute agent
        agent = AbducerAgent(db_session)
        result = await agent.run({
            "workspace_name": workspace.name,
            "observer": observer_peer.name,
            "observed": observed_peer.name,
        })

        # Should return with insufficient_premises reason
        assert result["hypotheses_created"] == 0
        assert result["hypothesis_ids"] == []
        assert result["reason"] == "insufficient_premises"

    @pytest.mark.asyncio
    async def test_retrieve_premises(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test premise retrieval filters by level and lookback period."""
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

        # Create explicit observations
        for i in range(5):
            doc = models.Document(
                workspace_name=workspace.name,
                observer=observer_peer.name,
                observed=observed_peer.name,
                content=f"Explicit observation {i}",
                embedding=[0.1 + i * 0.1] * 1536,
                session_name="test_session",
                level="explicit",
            )
            db_session.add(doc)

        # Create inductive observation (should be excluded)
        inductive_doc = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="Inductive observation",
            embedding=[0.9] * 1536,
            session_name="test_session",
            level="inductive",
        )
        db_session.add(inductive_doc)
        await db_session.flush()

        # Test premise retrieval
        agent = AbducerAgent(db_session)
        premises = await agent._retrieve_premises(
            workspace.name, observer_peer.name, observed_peer.name
        )

        # Should only return explicit observations
        assert len(premises) == 5
        assert all(p.level == "explicit" for p in premises)

    @pytest.mark.asyncio
    async def test_retrieve_existing_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test existing hypothesis retrieval."""
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

        # Create active hypothesis
        active_hyp = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="Active hypothesis",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
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
            ),
            workspace.name,
        )

        # Test retrieval
        agent = AbducerAgent(db_session)
        hypotheses = await agent._retrieve_existing_hypotheses(
            workspace.name, observer_peer.name, observed_peer.name
        )

        # Should only return active hypothesis
        assert len(hypotheses) == 1
        assert hypotheses[0].id == active_hyp.id
        assert hypotheses[0].status == "active"

    @pytest.mark.asyncio
    async def test_store_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test storing hypotheses in database."""
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

        # Prepare hypothesis data
        hypotheses_data = [
            {
                "content": "User prefers dark mode",
                "source_premise_ids": ["premise1", "premise2"],
                "confidence": 0.8,
                "tier": 0,
            },
            {
                "content": "User works late at night",
                "source_premise_ids": ["premise3"],
                "confidence": 0.7,
                "tier": 0,
            },
        ]

        # Store hypotheses
        agent = AbducerAgent(db_session)
        hypothesis_ids = await agent._store_hypotheses(
            hypotheses_data, workspace.name, observer_peer.name, observed_peer.name
        )

        # Verify storage
        assert len(hypothesis_ids) == 2

        # Retrieve and verify first hypothesis
        hyp1 = await crud.hypothesis.get_hypothesis(
            db_session, workspace.name, hypothesis_ids[0]
        )
        assert hyp1.content == "User prefers dark mode"
        assert hyp1.confidence == 0.8
        assert hyp1.source_premise_ids == ["premise1", "premise2"]
        assert hyp1.status == "active"
        assert hyp1.tier == 0

    @pytest.mark.asyncio
    async def test_custom_config(
        self,
        db_session: AsyncSession,
    ):
        """Test agent with custom configuration."""
        config = AbducerConfig(
            max_hypotheses_per_batch=10,
            min_premise_count=5,
            confidence_threshold=0.7,
            lookback_days=14,
        )

        agent = AbducerAgent(db_session, config=config)

        assert agent.config.max_hypotheses_per_batch == 10
        assert agent.config.min_premise_count == 5
        assert agent.config.confidence_threshold == 0.7
        assert agent.config.lookback_days == 14

    @pytest.mark.asyncio
    async def test_get_abducer_tools(
        self,
        db_session: AsyncSession,
    ):
        """Test tool definitions for abducer."""
        agent = AbducerAgent(db_session)
        tools = agent._get_abducer_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "create_hypothesis"
        assert "input_schema" in tools[0]

        # Verify required fields
        properties = tools[0]["input_schema"]["properties"]
        assert "content" in properties
        assert "source_premise_ids" in properties
        assert "confidence" in properties

        required = tools[0]["input_schema"]["required"]
        assert "content" in required
        assert "source_premise_ids" in required
        assert "confidence" in required
