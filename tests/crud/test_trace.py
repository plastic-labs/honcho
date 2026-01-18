"""Unit tests for FalsificationTrace CRUD operations."""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestTraceCRUD:
    """Test suite for falsification trace CRUD operations."""

    async def _create_test_prediction(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer_name: str,
        observed_name: str,
    ) -> str:
        """Helper to create a prediction (with hypothesis) and return its ID."""
        from nanoid import generate as generate_nanoid

        # Ensure collection exists
        await crud.collection.get_or_create_collection(
            db_session,
            workspace_name,
            observer=observer_name,
            observed=observed_name,
        )

        # Create hypothesis
        hypothesis_data = schemas.HypothesisCreate(
            content="Test hypothesis for traces",
            observer=observer_name,
            observed=observed_name,
        )
        hypothesis = await crud.hypothesis.create_hypothesis(
            db_session, hypothesis_data, workspace_name
        )

        # Create prediction
        prediction_data = schemas.PredictionCreate(
            content="Test prediction for traces",
            hypothesis_id=hypothesis.id,
        )
        prediction = await crud.prediction.create_prediction(
            db_session, prediction_data, workspace_name
        )
        return prediction.id

    @pytest.mark.asyncio
    async def test_create_trace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating a falsification trace."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        trace_data = schemas.FalsificationTraceCreate(
            prediction_id=prediction_id,
            search_queries=["query1", "query2"],
            contradicting_premise_ids=["premise1"],
            reasoning_chain={"step1": "search", "step2": "evaluate"},
            final_status="falsified",
            search_count=2,
            search_efficiency_score=0.85,
        )

        trace = await crud.trace.create_trace(
            db_session, trace_data, workspace.name
        )

        assert trace.id is not None
        assert len(trace.id) == 21
        assert trace.prediction_id == prediction_id
        assert trace.search_queries == ["query1", "query2"]
        assert trace.contradicting_premise_ids == ["premise1"]
        assert trace.reasoning_chain == {"step1": "search", "step2": "evaluate"}
        assert trace.final_status == "falsified"
        assert trace.search_count == 2
        assert trace.search_efficiency_score == 0.85
        assert trace.workspace_name == workspace.name
        assert trace.created_at is not None

    @pytest.mark.asyncio
    async def test_get_trace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving a trace by ID."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        trace_data = schemas.FalsificationTraceCreate(
            prediction_id=prediction_id,
            search_queries=["query1"],
            final_status="untested",
        )

        trace = await crud.trace.create_trace(
            db_session, trace_data, workspace.name
        )

        retrieved = await crud.trace.get_trace(
            db_session, workspace.name, trace.id
        )

        assert retrieved.id == trace.id
        assert retrieved.prediction_id == trace.prediction_id

    @pytest.mark.asyncio
    async def test_get_trace_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting a non-existent trace raises exception."""
        workspace, _ = sample_data

        with pytest.raises(ResourceNotFoundException):
            await crud.trace.get_trace(
                db_session, workspace.name, "nonexistent_id"
            )

    @pytest.mark.asyncio
    async def test_list_traces(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test listing traces with filtering."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create predictions for foreign keys
        prediction_ids = []
        for i in range(3):
            prediction_id = await self._create_test_prediction(
                db_session, workspace.name, observer_peer.name, observed_peer.name
            )
            prediction_ids.append(prediction_id)

        # Create multiple traces
        for i in range(3):
            trace_data = schemas.FalsificationTraceCreate(
                prediction_id=prediction_ids[i],
                search_queries=[f"query_{i}"],
                final_status="unfalsified" if i < 2 else "falsified",
                search_count=i + 1,
            )
            await crud.trace.create_trace(
                db_session, trace_data, workspace.name
            )

        # List all traces
        stmt = await crud.trace.list_traces(workspace_name=workspace.name)
        result = await db_session.execute(stmt)
        all_traces = list(result.scalars().all())

        assert len(all_traces) == 3

        # Filter by status
        stmt = await crud.trace.list_traces(
            workspace_name=workspace.name, final_status="unfalsified"
        )
        result = await db_session.execute(stmt)
        unfalsified_traces = list(result.scalars().all())

        assert len(unfalsified_traces) == 2

        # Filter by prediction_id
        stmt = await crud.trace.list_traces(
            workspace_name=workspace.name, prediction_id=prediction_ids[0]
        )
        result = await db_session.execute(stmt)
        prediction_traces = list(result.scalars().all())

        assert len(prediction_traces) == 1

    @pytest.mark.asyncio
    async def test_list_traces_by_date_range(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test filtering traces by date range."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        # Create traces
        trace_data = schemas.FalsificationTraceCreate(
            prediction_id=prediction_id,
            search_queries=["query"],
            final_status="untested",
        )

        await crud.trace.create_trace(db_session, trace_data, workspace.name)

        # Query by date range
        start_date = datetime.now(timezone.utc) - timedelta(hours=1)
        end_date = datetime.now(timezone.utc) + timedelta(hours=1)

        stmt = await crud.trace.list_traces(
            workspace_name=workspace.name,
            after_date=start_date,
            before_date=end_date,
        )
        result = await db_session.execute(stmt)
        traces = list(result.scalars().all())

        assert len(traces) == 1

    @pytest.mark.asyncio
    async def test_get_traces_by_prediction(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting all traces for a specific prediction."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        # Create multiple traces for the same prediction
        for i in range(3):
            trace_data = schemas.FalsificationTraceCreate(
                prediction_id=prediction_id,
                search_queries=[f"query_{i}"],
                final_status="untested",
            )
            await crud.trace.create_trace(
                db_session, trace_data, workspace.name
            )

        # Get all traces for this prediction
        traces = await crud.trace.get_traces_by_prediction(
            db_session, workspace.name, prediction_id
        )

        assert len(traces) == 3
        assert all(t.prediction_id == prediction_id for t in traces)
        # Should be ordered by creation time (ascending)
        assert traces[0].created_at <= traces[1].created_at <= traces[2].created_at

    @pytest.mark.asyncio
    async def test_trace_immutability(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that traces are immutable (no update operations)."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        trace_data = schemas.FalsificationTraceCreate(
            prediction_id=prediction_id,
            search_queries=["query"],
            final_status="untested",
        )

        trace = await crud.trace.create_trace(
            db_session, trace_data, workspace.name
        )

        # Verify no update function exists in crud.trace module
        assert not hasattr(crud.trace, "update_trace")
        # Verify trace was created
        assert trace.id is not None

    @pytest.mark.asyncio
    async def test_trace_defaults(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test trace creation with default values."""
        workspace, observer_peer = sample_data

        # Create observed peer
        from nanoid import generate as generate_nanoid
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add(observed_peer)
        await db_session.flush()

        # Create prediction for foreign key
        prediction_id = await self._create_test_prediction(
            db_session, workspace.name, observer_peer.name, observed_peer.name
        )

        trace_data = schemas.FalsificationTraceCreate(
            prediction_id=prediction_id,
        )

        trace = await crud.trace.create_trace(
            db_session, trace_data, workspace.name
        )

        # Check defaults
        assert trace.final_status == "untested"
        assert trace.search_count == 0
        assert trace.reasoning_chain == {}
