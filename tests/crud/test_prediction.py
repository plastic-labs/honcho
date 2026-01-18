"""Unit tests for Prediction CRUD operations."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestPredictionCRUD:
    """Test suite for prediction CRUD operations."""

    async def _create_test_hypothesis(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer_name: str,
        observed_name: str,
    ) -> str:
        """Helper to create a hypothesis and return its ID."""
        # Ensure collection exists
        await crud.collection.get_or_create_collection(
            db_session,
            workspace_name,
            observer=observer_name,
            observed=observed_name,
        )

        # Create hypothesis
        hypothesis_data = schemas.HypothesisCreate(
            content="Test hypothesis for predictions",
            observer=observer_name,
            observed=observed_name,
        )
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session, hypothesis_data, workspace_name
        )
        return hypothesis.id

    @pytest.mark.asyncio
    async def test_create_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating a prediction with vector embedding."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        prediction_data = schemas.PredictionCreate(
            content="The user will ask about dark mode",
            hypothesis_id=hypothesis_id,
            status="untested",
            is_blind=True,
        )

        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace.name
        )

        assert prediction.id is not None
        assert len(prediction.id) == 21
        assert prediction.content == "The user will ask about dark mode"
        assert prediction.hypothesis_id == hypothesis_id
        assert prediction.status == "untested"
        assert prediction.is_blind is True
        assert prediction.workspace_name == workspace.name
        assert prediction.created_at is not None
        # Embedding should be generated
        assert prediction.embedding is not None

    @pytest.mark.asyncio
    async def test_get_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving a prediction by ID."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        prediction_data = schemas.PredictionCreate(
            content="Test prediction",
            hypothesis_id=hypothesis_id,
        )

        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace.name
        )

        retrieved = await crud.prediction.get_prediction(
            db_session, workspace.name, prediction.id
        )

        assert retrieved.id == prediction.id
        assert retrieved.content == prediction.content

    @pytest.mark.asyncio
    async def test_get_prediction_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting a non-existent prediction raises exception."""
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException):
            await crud.prediction.get_prediction(
                db_session, workspace.name, "nonexistent_id"
            )

    @pytest.mark.asyncio
    async def test_update_prediction_status(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test updating prediction status (predictions are mostly immutable)."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        prediction_data = schemas.PredictionCreate(
            content="Test prediction",
            hypothesis_id=hypothesis_id,
            status="untested",
        )

        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace.name
        )

        # Update status to falsified
        update_data = schemas.PredictionUpdate(status="falsified")

        updated = await crud.prediction.update_prediction(
            db_session, workspace.name, prediction.id, update_data
        )

        assert updated.status == "falsified"
        # Content should remain unchanged
        assert updated.content == prediction.content

    @pytest.mark.asyncio
    async def test_delete_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test deleting a prediction."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        prediction_data = schemas.PredictionCreate(
            content="Test prediction",
            hypothesis_id=hypothesis_id,
        )

        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace.name
        )

        result = await crud.prediction.delete_prediction(
            db_session, workspace.name, prediction.id
        )

        assert result is True

        with pytest.raises(ResourceNotFoundException):
            await crud.prediction.get_prediction(
                db_session, workspace.name, prediction.id
            )

    @pytest.mark.asyncio
    async def test_list_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test listing predictions with filtering."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        # Create multiple predictions
        for i in range(3):
            prediction_data = schemas.PredictionCreate(
                content=f"Prediction {i}",
                hypothesis_id=hypothesis_id,
                status="untested" if i < 2 else "falsified",
                is_blind=(i % 2 == 0),
            )
            await crud.prediction.create_prediction(
                db_session, prediction_data, workspace.name
            )

        # List all predictions
        stmt = await crud.prediction.list_predictions(workspace_name=workspace.name)
        result = await db_session.execute(stmt)
        all_predictions = list(result.scalars().all())

        assert len(all_predictions) == 3

        # Filter by status
        stmt = await crud.prediction.list_predictions(
            workspace_name=workspace.name, status="untested"
        )
        result = await db_session.execute(stmt)
        untested_predictions = list(result.scalars().all())

        assert len(untested_predictions) == 2

        # Filter by hypothesis_id
        stmt = await crud.prediction.list_predictions(
            workspace_name=workspace.name, hypothesis_id=hypothesis_id
        )
        result = await db_session.execute(stmt)
        hypothesis_predictions = list(result.scalars().all())

        assert len(hypothesis_predictions) == 3

    @pytest.mark.asyncio
    async def test_search_predictions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test semantic search for predictions."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypotheses for foreign keys
        hypothesis_id_1 = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )
        hypothesis_id_2 = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        # Create predictions with different content
        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="User prefers dark mode",
                hypothesis_id=hypothesis_id_1,
            ),
            workspace.name,
        )

        await crud.prediction.create_prediction(
            db_session,
            schemas.PredictionCreate(
                content="User likes coffee",
                hypothesis_id=hypothesis_id_2,
            ),
            workspace.name,
        )

        # Search for predictions about preferences
        results = await crud.prediction.search_predictions(
            db_session, workspace.name, "user preferences", limit=10
        )

        # Should return results (order by similarity)
        assert len(results) > 0
        assert all(isinstance(p, models.Prediction) for p in results)

    @pytest.mark.asyncio
    async def test_prediction_defaults(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test prediction creation with default values."""
        workspace, observer_peer = sample_data

        # Create observed peer
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create hypothesis for foreign key
        hypothesis_id = await self._create_test_hypothesis(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        prediction_data = schemas.PredictionCreate(
            content="Minimal prediction",
            hypothesis_id=hypothesis_id,
        )

        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace.name
        )

        # Check defaults
        assert prediction.status == "untested"
        assert prediction.is_blind is True
