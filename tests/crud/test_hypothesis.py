"""Unit tests for Hypothesis CRUD operations."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestHypothesisCRUD:
    """Test suite for hypothesis CRUD operations."""

    async def _create_test_hypothesis(
        self,
        db_session: AsyncSession,
        workspace: models.Workspace,
        observer_peer: models.Peer,
        observed_peer: models.Peer,
    ) -> models.Hypothesis:
        """Helper to create a test hypothesis."""
        # Create collection (required for hypothesis foreign key)
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        hypothesis_data = schemas.HypothesisCreate(
            content="User prefers dark mode",
            observer=observer_peer.name,
            observed=observed_peer.name,
            status="active",
            confidence=0.8,
            source_premise_ids=["premise1", "premise2"],
            unaccounted_premises_count=5,
            search_coverage=10,
            tier=1,
        )

        return await crud.hypothesis.create_hypothesis(
            db_session, hypothesis_data, workspace.name
        )

    @pytest.mark.asyncio
    async def test_create_hypothesis(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating a hypothesis."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await self._create_test_hypothesis(
            db_session, workspace, observer_peer, observed_peer
        )

        assert hypothesis.id is not None
        assert len(hypothesis.id) == 21
        assert hypothesis.content == "User prefers dark mode"
        assert hypothesis.observer == observer_peer.name
        assert hypothesis.observed == observed_peer.name
        assert hypothesis.status == "active"
        assert hypothesis.confidence == 0.8
        assert hypothesis.source_premise_ids == ["premise1", "premise2"]
        assert hypothesis.unaccounted_premises_count == 5
        assert hypothesis.search_coverage == 10
        assert hypothesis.tier == 1
        assert hypothesis.workspace_name == workspace.name
        assert hypothesis.created_at is not None
        assert hypothesis.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_hypothesis(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving a hypothesis by ID."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await self._create_test_hypothesis(
            db_session, workspace, observer_peer, observed_peer
        )

        # Get hypothesis
        retrieved = await crud.hypothesis.get_hypothesis(
            db_session, workspace.name, hypothesis.id
        )

        assert retrieved.id == hypothesis.id
        assert retrieved.content == hypothesis.content
        assert retrieved.confidence == hypothesis.confidence

    @pytest.mark.asyncio
    async def test_get_hypothesis_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting a non-existent hypothesis raises exception."""
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException):
            await crud.hypothesis.get_hypothesis(
                db_session, workspace.name, "nonexistent_id"
            )

    @pytest.mark.asyncio
    async def test_update_hypothesis(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test updating a hypothesis."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await self._create_test_hypothesis(
            db_session, workspace, observer_peer, observed_peer
        )

        # Update hypothesis
        update_data = schemas.HypothesisUpdate(
            status="falsified",
            confidence=0.2,
            unaccounted_premises_count=2,
        )

        updated = await crud.hypothesis.update_hypothesis(
            db_session, workspace.name, hypothesis.id, update_data
        )

        assert updated.id == hypothesis.id
        assert updated.status == "falsified"
        assert updated.confidence == 0.2
        assert updated.unaccounted_premises_count == 2
        # Original content should remain unchanged
        assert updated.content == hypothesis.content

    @pytest.mark.asyncio
    async def test_delete_hypothesis(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test deleting a hypothesis."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis
        hypothesis = await self._create_test_hypothesis(
            db_session, workspace, observer_peer, observed_peer
        )

        # Delete hypothesis
        result = await crud.hypothesis.delete_hypothesis(
            db_session, workspace.name, hypothesis.id
        )

        assert result is True

        # Verify deleted
        with pytest.raises(ResourceNotFoundException):
            await crud.hypothesis.get_hypothesis(
                db_session, workspace.name, hypothesis.id
            )

    @pytest.mark.asyncio
    async def test_list_hypotheses(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test listing hypotheses with filtering."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection (required for hypothesis foreign key)
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create multiple hypotheses
        for i in range(3):
            hypothesis_data = schemas.HypothesisCreate(
                content=f"Hypothesis {i}",
                observer=observer_peer.name,
                observed=observed_peer.name,
                status="active" if i < 2 else "falsified",
            )
            await crud.hypothesis.create_hypothesis(
                db_session, hypothesis_data, workspace.name
            )

        # List all hypotheses
        stmt = await crud.hypothesis.list_hypotheses(workspace_name=workspace.name)
        result = await db_session.execute(stmt)
        all_hypotheses = list(result.scalars().all())

        assert len(all_hypotheses) == 3

        # Filter by status
        stmt = await crud.hypothesis.list_hypotheses(
            workspace_name=workspace.name, status="active"
        )
        result = await db_session.execute(stmt)
        active_hypotheses = list(result.scalars().all())

        assert len(active_hypotheses) == 2

        # Filter by observer
        stmt = await crud.hypothesis.list_hypotheses(
            workspace_name=workspace.name, observer=observer_peer.name
        )
        result = await db_session.execute(stmt)
        observer_hypotheses = list(result.scalars().all())

        assert len(observer_hypotheses) == 3

    @pytest.mark.asyncio
    async def test_hypothesis_defaults(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test hypothesis creation with default values."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection (required for hypothesis foreign key)
        await crud.collection.get_or_create_collection(
            db_session,
            workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Create minimal hypothesis
        hypothesis_data = schemas.HypothesisCreate(
            content="Minimal hypothesis",
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session, hypothesis_data, workspace.name
        )

        # Check defaults
        assert hypothesis.status == "active"
        assert hypothesis.confidence == 0.5
        assert hypothesis.unaccounted_premises_count == 0
        assert hypothesis.search_coverage == 0
        assert hypothesis.tier == 0
        assert hypothesis.reasoning_metadata == {}
