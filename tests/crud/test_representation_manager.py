import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.representation import RepresentationManager
from src.schemas import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.representation import ExplicitObservation, Representation


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


class TestRepresentationManagerSaveFiltering:
    """Tests that save_representation filters invalid observations before embedding."""

    @pytest.mark.asyncio
    async def test_empty_content_observations_are_filtered_before_embedding(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """Observations with empty or whitespace-only content must be dropped before
        the embedding API call. OpenAI returns 400 Bad Request for empty strings.
        """
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=generate_nanoid(), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        test_session = models.Session(
            name=generate_nanoid(), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        manager = RepresentationManager(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        captured_texts: list[list[str]] = []

        async def fake_batch_embed(texts: list[str]) -> list[list[float]]:
            captured_texts.append(list(texts))
            return [[0.1] * 8 for _ in texts]

        @asynccontextmanager
        async def fake_tracked_db(*args: object, **kwargs: object):  # type: ignore[misc]
            yield db_session

        now = datetime.datetime.now(datetime.timezone.utc)
        config = ResolvedConfiguration(
            reasoning=ResolvedReasoningConfiguration(enabled=False),
            peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
            summary=ResolvedSummaryConfiguration(
                enabled=False,
                messages_per_short_summary=20,
                messages_per_long_summary=60,
            ),
            dream=ResolvedDreamConfiguration(enabled=False),
        )
        rep = Representation(
            explicit=[
                ExplicitObservation(
                    content="valid observation",
                    created_at=now,
                    message_ids=[1],
                    session_name=test_session.name,
                ),
                ExplicitObservation(
                    content="",
                    created_at=now,
                    message_ids=[1],
                    session_name=test_session.name,
                ),
                ExplicitObservation(
                    content="   ",
                    created_at=now,
                    message_ids=[1],
                    session_name=test_session.name,
                ),
                ExplicitObservation(
                    content="another valid",
                    created_at=now,
                    message_ids=[1],
                    session_name=test_session.name,
                ),
            ]
        )

        with (
            patch(
                "src.crud.representation.embedding_client.simple_batch_embed",
                fake_batch_embed,
            ),
            patch("src.crud.representation.tracked_db", fake_tracked_db),
            patch(
                "src.crud.representation.check_and_schedule_dream", AsyncMock()
            ),
            patch("src.crud.representation.accumulate_metric"),
        ):
            count = await manager.save_representation(
                rep,
                message_ids=[1],
                session_name=test_session.name,
                message_created_at=now,
                message_level_configuration=config,
            )

        assert len(captured_texts) == 1, "embedding client called exactly once"
        assert captured_texts[0] == [
            "valid observation",
            "another valid",
        ], "only non-empty texts passed to embedder"
        assert count == 2, "two documents saved for the two non-empty observations"

    @pytest.mark.asyncio
    async def test_all_empty_content_observations_skips_embedding(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """If every observation has empty content, embedding is never called and 0 is returned."""
        test_workspace, test_peer = sample_data

        test_peer2 = models.Peer(
            name=generate_nanoid(), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        manager = RepresentationManager(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        embed_called = False

        async def fake_batch_embed(texts: list[str]) -> list[list[float]]:  # pragma: no cover
            nonlocal embed_called
            embed_called = True
            return []

        now = datetime.datetime.now(datetime.timezone.utc)
        config = ResolvedConfiguration(
            reasoning=ResolvedReasoningConfiguration(enabled=False),
            peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
            summary=ResolvedSummaryConfiguration(
                enabled=False,
                messages_per_short_summary=20,
                messages_per_long_summary=60,
            ),
            dream=ResolvedDreamConfiguration(enabled=False),
        )
        rep = Representation(
            explicit=[
                ExplicitObservation(
                    content="",
                    created_at=now,
                    message_ids=[1],
                    session_name="s",
                ),
                ExplicitObservation(
                    content="  ",
                    created_at=now,
                    message_ids=[1],
                    session_name="s",
                ),
            ]
        )

        with patch(
            "src.crud.representation.embedding_client.simple_batch_embed",
            fake_batch_embed,
        ):
            count = await manager.save_representation(
                rep,
                message_ids=[1],
                session_name="s",
                message_created_at=now,
                message_level_configuration=config,
            )

        assert not embed_called, "embedding client must not be called for all-empty input"
        assert count == 0
