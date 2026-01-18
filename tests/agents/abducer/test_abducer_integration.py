"""Integration tests for the Abducer agent."""

from typing import Any
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.agents.abducer import AbducerAgent, AbducerConfig


class TestAbducerIntegration:
    """Integration test suite for Abducer agent end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_hypothesis_generation(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test complete hypothesis generation workflow from premises to storage."""
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

        # Create sufficient premises for hypothesis generation
        premise_contents = [
            "User prefers dark mode in the evening",
            "User works late at night frequently",
            "User mentions eye strain from bright screens",
            "User has reduced screen brightness to minimum",
            "User uses blue light filter on devices",
        ]

        for i, content in enumerate(premise_contents):
            doc = models.Document(
                workspace_name=workspace.name,
                observer=observer_peer.name,
                observed=observed_peer.name,
                content=content,
                embedding=[0.1 + i * 0.05] * 1536,
                session_name="test_session",
                level="explicit",
            )
            db_session.add(doc)
        await db_session.flush()

        # Mock LLM call to simulate tool execution
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # Get recent docs for source_premise_ids
                recent_docs = await self._get_recent_docs(db_session, workspace.name)

                # Simulate tool calls
                tool_executor("create_hypothesis", {
                    "content": "User is sensitive to bright light and prefers low-light environments",
                    "source_premise_ids": [doc.id for doc in recent_docs[:3]],
                    "confidence": 0.85,
                    "tier": 0,
                })
                tool_executor("create_hypothesis", {
                    "content": "User experiences screen-related eye strain and actively mitigates it",
                    "source_premise_ids": [doc.id for doc in recent_docs[:5]],
                    "confidence": 0.80,
                    "tier": 0,
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Generated 2 hypotheses based on the observations."
            return mock_response

        # Patch honcho_llm_call to use mock
        with patch("src.agents.abducer.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent
            agent = AbducerAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Verify results
        assert result["hypotheses_created"] == 2
        assert len(result["hypothesis_ids"]) == 2

        # Verify hypotheses were stored in database
        hypothesis1 = await crud.hypothesis.get_hypothesis(
            db_session, workspace.name, result["hypothesis_ids"][0]
        )
        assert hypothesis1 is not None
        assert "sensitive to bright light" in hypothesis1.content
        assert hypothesis1.confidence == 0.85
        assert hypothesis1.status == "active"
        assert hypothesis1.tier == 0

        hypothesis2 = await crud.hypothesis.get_hypothesis(
            db_session, workspace.name, result["hypothesis_ids"][1]
        )
        assert hypothesis2 is not None
        assert "eye strain" in hypothesis2.content
        assert hypothesis2.confidence == 0.80

    @pytest.mark.asyncio
    async def test_hypothesis_generation_with_existing_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that agent considers existing hypotheses to avoid duplicates."""
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

        # Create existing hypothesis
        _existing_hypothesis = await crud.hypothesis.create_hypothesis(
            db_session,
            schemas.HypothesisCreate(
                content="User prefers dark mode",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active",
                confidence=0.75,
                source_premise_ids=["doc1", "doc2"],
                tier=0,
            ),
            workspace.name,
        )

        # Create sufficient premises
        for i in range(5):
            doc = models.Document(
                workspace_name=workspace.name,
                observer=observer_peer.name,
                observed=observed_peer.name,
                content=f"Observation about dark mode {i}",
                embedding=[0.2 + i * 0.1] * 1536,
                session_name="test_session",
                level="explicit",
            )
            db_session.add(doc)
        await db_session.flush()

        # Mock LLM call - agent should see existing hypothesis
        mock_llm_response = MagicMock()
        mock_llm_response.tool_calls_made = []
        mock_llm_response.text = "No new hypotheses needed - existing hypothesis covers the observations."

        with patch("src.agents.abducer.agent.honcho_llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            # Execute agent
            agent = AbducerAgent(db_session)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

            # Verify LLM was called with existing hypothesis in context
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args
            messages = call_args.kwargs["messages"]
            task_prompt = messages[1]["content"]

            # Existing hypothesis should be mentioned in prompt
            assert "User prefers dark mode" in task_prompt
            assert "0.75" in task_prompt

        # No new hypotheses created
        assert result["hypotheses_created"] == 0

    @pytest.mark.asyncio
    async def test_hypothesis_generation_with_custom_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test hypothesis generation respects custom configuration."""
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

        # Custom config with higher confidence threshold
        config = AbducerConfig(
            min_premise_count=2,
            confidence_threshold=0.9,
            max_hypotheses_per_batch=3,
        )

        # Create sufficient premises
        for i in range(5):
            doc = models.Document(
                workspace_name=workspace.name,
                observer=observer_peer.name,
                observed=observed_peer.name,
                content=f"Important observation {i}",
                embedding=[0.3 + i * 0.1] * 1536,
                session_name="test_session",
                level="explicit",
            )
            db_session.add(doc)
        await db_session.flush()

        # Mock LLM call to simulate tool execution with confidence threshold
        async def mock_llm_call(*_args: Any, **kwargs: Any) -> MagicMock:
            """Mock LLM that executes the tool_executor."""
            tool_executor = kwargs.get("tool_executor")  # type: ignore[reportUnknownMemberType]
            if tool_executor:
                # Try to create hypotheses with different confidence levels
                # Low confidence should be rejected
                tool_executor("create_hypothesis", {
                    "content": "Low confidence hypothesis",
                    "source_premise_ids": ["doc1", "doc2"],
                    "confidence": 0.75,  # Below threshold (0.9)
                    "tier": 0,
                })

                # High confidence should be accepted
                tool_executor("create_hypothesis", {
                    "content": "High confidence hypothesis",
                    "source_premise_ids": ["doc1", "doc2"],
                    "confidence": 0.95,  # Above threshold
                    "tier": 0,
                })

            mock_response = MagicMock()
            mock_response.tool_calls_made = ["call1", "call2"]
            mock_response.text = "Generated 1 high-confidence hypothesis."
            return mock_response

        with patch("src.agents.abducer.agent.honcho_llm_call", side_effect=mock_llm_call):
            # Execute agent with custom config
            agent = AbducerAgent(db_session, config=config)
            result = await agent.run({
                "workspace_name": workspace.name,
                "observer": observer_peer.name,
                "observed": observed_peer.name,
            })

        # Only 1 hypothesis should be created (high confidence one)
        assert result["hypotheses_created"] == 1

        hypothesis = await crud.hypothesis.get_hypothesis(
            db_session, workspace.name, result["hypothesis_ids"][0]
        )
        assert hypothesis.confidence == 0.95

    async def _get_recent_docs(
        self, db_session: AsyncSession, workspace_name: str
    ) -> list[models.Document]:
        """Helper to retrieve recent documents."""
        from sqlalchemy import select

        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_name)
            .order_by(models.Document.created_at.desc())
        )
        result = await db_session.execute(stmt)
        return list(result.scalars().all())
