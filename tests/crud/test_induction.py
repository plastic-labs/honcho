"""Unit tests for Induction CRUD operations."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestInductionCRUD:
    """Test suite for induction CRUD operations."""

    async def _ensure_collection(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer_name: str,
        observed_name: str,
    ) -> None:
        """Helper to ensure collection exists for foreign key."""
        await crud.collection.get_or_create_collection(
            db_session,
            workspace_name,
            observer=observer_name,
            observed=observed_name,
        )

    @pytest.mark.asyncio
    async def test_create_induction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating an induction."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create collection (required for induction foreign key)
        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        induction_data = schemas.InductionCreate(
            content="Users typically prefer dark mode at night",
            observer=observer_peer.name,
            observed=observed_peer.name,
            pattern_type="temporal",
            source_prediction_ids=["pred1", "pred2"],
            source_premise_ids=["prem1", "prem2"],
            confidence="high",
            stability_score=0.9,
        )

        induction = await crud.induction.create_induction(
            db_session, induction_data, workspace.name
        )

        assert induction.id is not None
        assert len(induction.id) == 21
        assert induction.content == "Users typically prefer dark mode at night"
        assert induction.observer == observer_peer.name
        assert induction.observed == observed_peer.name
        assert induction.pattern_type == "temporal"
        assert induction.source_prediction_ids == ["pred1", "pred2"]
        assert induction.source_premise_ids == ["prem1", "prem2"]
        assert induction.confidence == "high"
        assert induction.stability_score == 0.9
        assert induction.workspace_name == workspace.name
        assert induction.created_at is not None

    @pytest.mark.asyncio
    async def test_get_induction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving an induction by ID."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        induction_data = schemas.InductionCreate(
            content="Test induction",
            observer=observer_peer.name,
            observed=observed_peer.name,
            pattern_type="preference",
        )

        induction = await crud.induction.create_induction(
            db_session, induction_data, workspace.name
        )

        retrieved = await crud.induction.get_induction(
            db_session, workspace.name, induction.id
        )

        assert retrieved.id == induction.id
        assert retrieved.content == induction.content
        assert retrieved.pattern_type == "preference"

    @pytest.mark.asyncio
    async def test_get_induction_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting a non-existent induction raises exception."""
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException):
            await crud.induction.get_induction(
                db_session, workspace.name, "nonexistent_id"
            )

    @pytest.mark.asyncio
    async def test_delete_induction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test deleting an induction."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        induction_data = schemas.InductionCreate(
            content="Test induction",
            observer=observer_peer.name,
            observed=observed_peer.name,
            pattern_type="personality",
        )

        induction = await crud.induction.create_induction(
            db_session, induction_data, workspace.name
        )

        result = await crud.induction.delete_induction(
            db_session, workspace.name, induction.id
        )

        assert result is True

        with pytest.raises(ResourceNotFoundException):
            await crud.induction.get_induction(
                db_session, workspace.name, induction.id
            )

    @pytest.mark.asyncio
    async def test_list_inductions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test listing inductions with filtering."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        # Create multiple inductions
        pattern_types = ["temporal", "behavior", "conditional"]
        confidences = ["high", "medium", "low"]

        for i in range(3):
            induction_data = schemas.InductionCreate(
                content=f"Induction {i}",
                observer=observer_peer.name,
                observed=observed_peer.name,
                pattern_type=pattern_types[i],
                confidence=confidences[i],
            )
            await crud.induction.create_induction(
                db_session, induction_data, workspace.name
            )

        # List all inductions
        stmt = await crud.induction.list_inductions(workspace_name=workspace.name)
        result = await db_session.execute(stmt)
        all_inductions = list(result.scalars().all())

        assert len(all_inductions) == 3

        # Filter by pattern_type
        stmt = await crud.induction.list_inductions(
            workspace_name=workspace.name, pattern_type="temporal"
        )
        result = await db_session.execute(stmt)
        temporal_inductions = list(result.scalars().all())

        assert len(temporal_inductions) == 1
        assert temporal_inductions[0].pattern_type == "temporal"

        # Filter by confidence
        stmt = await crud.induction.list_inductions(
            workspace_name=workspace.name, confidence="high"
        )
        result = await db_session.execute(stmt)
        high_confidence_inductions = list(result.scalars().all())

        assert len(high_confidence_inductions) == 1
        assert high_confidence_inductions[0].confidence == "high"

        # Filter by observer
        stmt = await crud.induction.list_inductions(
            workspace_name=workspace.name, observer=observer_peer.name
        )
        result = await db_session.execute(stmt)
        observer_inductions = list(result.scalars().all())

        assert len(observer_inductions) == 3

    @pytest.mark.asyncio
    async def test_induction_pattern_types(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test all induction pattern types."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        pattern_types = ["temporal", "behavior", "conditional", "tendency"]

        for pattern_type in pattern_types:
            induction_data = schemas.InductionCreate(
                content=f"Pattern: {pattern_type}",
                observer=observer_peer.name,
                observed=observed_peer.name,
                pattern_type=pattern_type,
            )

            induction = await crud.induction.create_induction(
                db_session, induction_data, workspace.name
            )

            assert induction.pattern_type == pattern_type

    @pytest.mark.asyncio
    async def test_induction_confidence_levels(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test all induction confidence levels."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        confidence_levels = ["high", "medium", "low"]

        for confidence in confidence_levels:
            induction_data = schemas.InductionCreate(
                content=f"Confidence: {confidence}",
                observer=observer_peer.name,
                observed=observed_peer.name,
                pattern_type="temporal",
                confidence=confidence,
            )

            induction = await crud.induction.create_induction(
                db_session, induction_data, workspace.name
            )

            assert induction.confidence == confidence

    @pytest.mark.asyncio
    async def test_induction_defaults(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test induction creation with default values."""
        workspace, observer_peer = sample_data

        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        await self._ensure_collection(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        induction_data = schemas.InductionCreate(
            content="Minimal induction",
            observer=observer_peer.name,
            observed=observed_peer.name,
            pattern_type="tendency",
        )

        induction = await crud.induction.create_induction(
            db_session, induction_data, workspace.name
        )

        # Check defaults
        assert induction.confidence == "medium"
        assert induction.stability_score is None
