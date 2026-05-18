from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.representation import RepresentationManager
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)


@asynccontextmanager
async def _fake_tracked_db(_name: str):
    yield object()


def _saved_observations(mock_save: AsyncMock):
    call = mock_save.await_args
    assert call is not None, "mock was not awaited"
    if "all_observations" in call.kwargs:
        return call.kwargs["all_observations"]
    if len(call.args) > 1:
        return call.args[1]
    raise AssertionError("missing all_observations in await args")


class TestRepresentationManagerSoftDelete:
    """Tests that RepresentationManager query methods exclude soft-deleted documents."""

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Collection, RepresentationManager]:
        """Create peers, session, collection, and a RepresentationManager."""
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        manager = RepresentationManager(
            test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        return test_peer2, test_session, collection, manager

    @pytest.mark.asyncio
    async def test_query_documents_recent_excludes_soft_deleted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Soft-deleted documents must not appear in the recent-documents query."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _, manager = await self._setup(
            db_session, test_workspace, test_peer
        )

        # Create two documents
        doc_live = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Live observation",
            session_name=test_session.name,
        )
        doc_deleted = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Deleted observation",
            session_name=test_session.name,
        )
        db_session.add_all([doc_live, doc_deleted])
        await db_session.flush()

        # Soft-delete one
        await db_session.execute(
            update(models.Document)
            .where(models.Document.id == doc_deleted.id)
            .values(deleted_at=func.now())
        )
        await db_session.commit()

        results = await manager._query_documents_recent(db_session, top_k=10)  # pyright: ignore[reportPrivateUsage]

        result_ids = [doc.id for doc in results]
        assert doc_live.id in result_ids
        assert doc_deleted.id not in result_ids

    @pytest.mark.asyncio
    async def test_query_documents_most_derived_excludes_soft_deleted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Soft-deleted documents must not appear in the most-derived query."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _, manager = await self._setup(
            db_session, test_workspace, test_peer
        )

        # Create two documents with different times_derived
        doc_live = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Live observation",
            session_name=test_session.name,
            times_derived=5,
        )
        doc_deleted = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Deleted high-derived observation",
            session_name=test_session.name,
            times_derived=100,
        )
        db_session.add_all([doc_live, doc_deleted])
        await db_session.flush()

        # Soft-delete the high-derived one
        await db_session.execute(
            update(models.Document)
            .where(models.Document.id == doc_deleted.id)
            .values(deleted_at=func.now())
        )
        await db_session.commit()

        results = await manager._query_documents_most_derived(db_session, top_k=10)  # pyright: ignore[reportPrivateUsage]

        result_ids = [doc.id for doc in results]
        assert doc_live.id in result_ids
        assert doc_deleted.id not in result_ids


class TestRepresentationManagerSave:
    @pytest.mark.asyncio
    async def test_save_representation_filters_blank_observations_before_embedding(
        self,
    ):
        manager = RepresentationManager(
            "workspace",
            observer="observer",
            observed="observed",
        )
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="   ",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
                ExplicitObservation(
                    content=" useful observation ",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
            ]
        )

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.embedding_client.simple_batch_embed",
                new=AsyncMock(return_value=[[0.1]]),
            ) as mock_embed,
            patch.object(
                manager,
                "_save_representation_internal",
                new=AsyncMock(return_value=1),
            ) as mock_save,
        ):
            saved = await manager.save_representation(
                representation,
                message_ids=[1],
                session_name="session",
                message_created_at=datetime.now(timezone.utc),
                message_level_configuration=SimpleNamespace(  # pyright: ignore[reportArgumentType]
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 1
        mock_embed.assert_awaited_once_with(["useful observation"])
        saved_observations = _saved_observations(mock_save)
        assert len(saved_observations) == 1
        assert saved_observations[0].content == "useful observation"

    @pytest.mark.asyncio
    async def test_save_representation_filters_blank_deductive_observations(self):
        manager = RepresentationManager(
            "workspace",
            observer="observer",
            observed="observed",
        )
        representation = Representation(
            deductive=[
                DeductiveObservation(
                    conclusion="   ",
                    premises=["premise a"],
                    source_ids=["doc-a"],
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
                DeductiveObservation(
                    conclusion=" inferred conclusion ",
                    premises=["premise b"],
                    source_ids=["doc-b"],
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
            ]
        )

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.embedding_client.simple_batch_embed",
                new=AsyncMock(return_value=[[0.2]]),
            ) as mock_embed,
            patch.object(
                manager,
                "_save_representation_internal",
                new=AsyncMock(return_value=1),
            ) as mock_save,
        ):
            saved = await manager.save_representation(
                representation,
                message_ids=[1],
                session_name="session",
                message_created_at=datetime.now(timezone.utc),
                message_level_configuration=SimpleNamespace(  # pyright: ignore[reportArgumentType]
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 1
        mock_embed.assert_awaited_once_with(["inferred conclusion"])
        saved_observations = _saved_observations(mock_save)
        assert len(saved_observations) == 1
        assert isinstance(saved_observations[0], DeductiveObservation)
        assert saved_observations[0].conclusion == "inferred conclusion"

    @pytest.mark.asyncio
    async def test_save_representation_skips_all_blank_observations(self):
        manager = RepresentationManager(
            "workspace",
            observer="observer",
            observed="observed",
        )
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
                ExplicitObservation(
                    content="\n\t ",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[1],
                    session_name="session",
                ),
            ]
        )

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.embedding_client.simple_batch_embed",
                new=AsyncMock(),
            ) as mock_embed,
            patch.object(
                manager,
                "_save_representation_internal",
                new=AsyncMock(),
            ) as mock_save,
        ):
            saved = await manager.save_representation(
                representation,
                message_ids=[1],
                session_name="session",
                message_created_at=datetime.now(timezone.utc),
                message_level_configuration=SimpleNamespace(  # pyright: ignore[reportArgumentType]
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 0
        mock_embed.assert_not_awaited()
        mock_save.assert_not_awaited()
