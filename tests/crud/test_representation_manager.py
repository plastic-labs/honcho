from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.document import CreateDocumentsResult
from src.crud.representation import RepresentationManager
from src.schemas.configuration import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)


def _resolved_config(*, dream_enabled: bool = False) -> ResolvedConfiguration:
    """Build a minimal ResolvedConfiguration for tests that only care about dream.enabled."""
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=False),
        peer_card=ResolvedPeerCardConfiguration(use=False, create=False),
        summary=ResolvedSummaryConfiguration(
            enabled=False,
            messages_per_short_summary=20,
            messages_per_long_summary=60,
        ),
        dream=ResolvedDreamConfiguration(enabled=dream_enabled),
    )


@asynccontextmanager
async def _fake_tracked_db(_name: str):
    yield object()


def _saved_observations(mock_save: AsyncMock):
    call = mock_save.await_args
    assert call is not None, "mock_save was never awaited"
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

    @pytest.mark.asyncio
    async def test_query_documents_most_derived_ties_break_by_recency(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Regression: when times_derived ties, the manager's most-derived query
        must fall back to recency, not insertion order. Mirrors the equivalent
        test on crud.query_documents_most_derived -- the query is duplicated in
        both modules and must not drift."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _, manager = await self._setup(
            db_session, test_workspace, test_peer
        )

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Three conclusions, all reinforced once, inserted oldest-first.
        for i in range(3):
            db_session.add(
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content=f"tie {i}",
                    session_name=test_session.name,
                    times_derived=1,
                    created_at=base + timedelta(days=i),
                )
            )
        # A genuinely reinforced conclusion that is also the oldest of all.
        db_session.add(
            models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content="hot",
                session_name=test_session.name,
                times_derived=5,
                created_at=base - timedelta(days=10),
            )
        )
        await db_session.flush()

        results = await manager._query_documents_most_derived(db_session, top_k=10)  # pyright: ignore[reportPrivateUsage]

        contents = [doc.content for doc in results]
        # Primary sort still wins: the actually-reinforced conclusion leads.
        assert contents[0] == "hot"
        # Ties break toward most-recent, not oldest-inserted.
        assert contents[1:] == ["tie 2", "tie 1", "tie 0"]


class TestRepresentationManagerSessionScoping:
    """Tests that the session allowlist is applied uniformly to every query path.

    Regression for DEV-1994: session_name used to be applied only to the
    recent-documents query; the semantic and most-derived paths ignored it,
    so limit_to_session leaked cross-session conclusions.
    """

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Session, models.Session, RepresentationManager]:
        """Create two sessions and documents in each, plus a session-less doc."""
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([session_a, session_b])
        await db_session.flush()

        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        db_session.add_all(
            [
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content="in-scope observation",
                    session_name=session_a.name,
                    times_derived=1,
                ),
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content="out-of-scope observation",
                    session_name=session_b.name,
                    times_derived=100,
                ),
                # Dream-produced documents have no session_name; a session
                # allowlist must exclude them (fail-closed).
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content="sessionless dream observation",
                    session_name=None,
                    times_derived=50,
                ),
            ]
        )
        await db_session.flush()

        manager = RepresentationManager(
            test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        return session_a, session_b, manager

    @pytest.mark.asyncio
    async def test_recent_respects_session_allowlist(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        session_a, _, manager = await self._setup(db_session, test_workspace, test_peer)

        results = await manager._query_documents_recent(  # pyright: ignore[reportPrivateUsage]
            db_session, top_k=10, session_names=[session_a.name]
        )

        contents = [doc.content for doc in results]
        assert contents == ["in-scope observation"]

    @pytest.mark.asyncio
    async def test_most_derived_respects_session_allowlist(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """The out-of-scope doc has far higher times_derived; it must still be excluded."""
        test_workspace, test_peer = sample_data
        session_a, _, manager = await self._setup(db_session, test_workspace, test_peer)

        results = await manager._query_documents_most_derived(  # pyright: ignore[reportPrivateUsage]
            db_session, top_k=10, session_names=[session_a.name]
        )

        contents = [doc.content for doc in results]
        assert contents == ["in-scope observation"]

    @pytest.mark.asyncio
    async def test_semantic_passes_session_allowlist_as_filters(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """The semantic path must push the allowlist down to query_documents."""
        test_workspace, test_peer = sample_data
        session_a, _, manager = await self._setup(db_session, test_workspace, test_peer)

        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await manager._query_documents_semantic(  # pyright: ignore[reportPrivateUsage]
                db_session,
                query="anything",
                top_k=5,
                embedding=[0.1],
                session_names=[session_a.name],
            )

        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {
            "session_name": {"in": [session_a.name]}
        }

    @pytest.mark.asyncio
    async def test_semantic_passes_no_filters_when_unscoped(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        _, _, manager = await self._setup(db_session, test_workspace, test_peer)

        with patch(
            "src.crud.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            await manager._query_documents_semantic(  # pyright: ignore[reportPrivateUsage]
                db_session,
                query="anything",
                top_k=5,
                embedding=[0.1],
            )

        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] is None

    @pytest.mark.asyncio
    async def test_working_representation_scoped_end_to_end(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """All blended paths active: only in-scope content may appear."""
        test_workspace, test_peer = sample_data
        session_a, _, manager = await self._setup(db_session, test_workspace, test_peer)

        representation = await manager.get_working_representation(
            db=db_session,
            session_names=[session_a.name],
            include_most_derived=True,
        )

        contents = [obs.content for obs in representation.explicit]
        assert "in-scope observation" in contents
        assert "out-of-scope observation" not in contents
        assert "sessionless dream observation" not in contents

    @pytest.mark.asyncio
    async def test_empty_allowlist_fails_closed(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """An empty allowlist must return an empty representation, not fall
        back to unscoped behavior (downstream stores drop empty IN clauses)."""
        test_workspace, test_peer = sample_data
        _, _, manager = await self._setup(db_session, test_workspace, test_peer)

        representation = await manager.get_working_representation(
            db=db_session,
            session_names=[],
            include_most_derived=True,
        )

        assert representation.explicit == []
        assert representation.deductive == []

    def test_build_filter_conditions_empty_allowlist_fails_closed(self):
        """The filter-builder layer itself must fail closed, independent of the
        early-return guard in _get_working_representation_internal. An empty
        allowlist emits an empty `in` (renders as always-false downstream), not
        an omitted filter."""
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )

        assert manager._build_filter_conditions(session_names=[]) == {  # pyright: ignore[reportPrivateUsage]
            "session_name": {"in": []}
        }
        # None means unscoped — no session filter emitted.
        assert manager._build_filter_conditions(session_names=None) == {}  # pyright: ignore[reportPrivateUsage]
        assert manager._build_filter_conditions(session_names=["s1"]) == {  # pyright: ignore[reportPrivateUsage]
            "session_name": {"in": ["s1"]}
        }


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
                new=AsyncMock(
                    return_value=CreateDocumentsResult(created_documents=[MagicMock()])
                ),
            ) as mock_save,
        ):
            saved = await manager.save_representation(
                representation,
                message_ids=[1],
                session_name="session",
                message_created_at=datetime.now(timezone.utc),
                message_level_configuration=_resolved_config(),
            )

        assert len(saved.created_documents) == 1
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
                new=AsyncMock(
                    return_value=CreateDocumentsResult(created_documents=[MagicMock()])
                ),
            ) as mock_save,
        ):
            saved = await manager.save_representation(
                representation,
                message_ids=[1],
                session_name="session",
                message_created_at=datetime.now(timezone.utc),
                message_level_configuration=_resolved_config(),
            )

        assert len(saved.created_documents) == 1
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
                message_level_configuration=_resolved_config(),
            )

        assert len(saved.created_documents) == 0
        mock_embed.assert_not_awaited()
        mock_save.assert_not_awaited()
