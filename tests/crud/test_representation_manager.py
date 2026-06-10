from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
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
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=0),
            ),
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
                message_level_configuration=_resolved_config(),
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
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=0),
            ),
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
                message_level_configuration=_resolved_config(),
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
                message_level_configuration=_resolved_config(),
            )

        assert saved == 0
        mock_embed.assert_not_awaited()
        mock_save.assert_not_awaited()


class TestRepresentationManagerSessionCap:
    """Tests for the per-session observation burst cap."""

    @staticmethod
    def _explicit(content: str) -> ExplicitObservation:
        return ExplicitObservation(
            content=content,
            created_at=datetime.now(timezone.utc),
            message_ids=[1],
            session_name="session",
        )

    @pytest.mark.asyncio
    async def test_cap_truncates_burst_to_headroom(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A batch overflowing the cap is trimmed to the remaining headroom."""
        monkeypatch.setattr(settings.DERIVER, "MAX_OBSERVATIONS_PER_SESSION", 50)
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )
        observations: list[ExplicitObservation | DeductiveObservation] = [
            self._explicit(f"obs {i}") for i in range(5)
        ]

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=48),
            ),
        ):
            result = await manager._apply_session_observation_cap(  # pyright: ignore[reportPrivateUsage]
                observations, "session"
            )

        # headroom = 50 - 48 = 2
        assert [o.content for o in result if isinstance(o, ExplicitObservation)] == [
            "obs 0",
            "obs 1",
        ]

    @pytest.mark.asyncio
    async def test_cap_drops_entire_batch_when_session_full(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When the session is already at/over the cap, the whole batch is dropped."""
        monkeypatch.setattr(settings.DERIVER, "MAX_OBSERVATIONS_PER_SESSION", 50)
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )
        observations: list[ExplicitObservation | DeductiveObservation] = [
            self._explicit(f"obs {i}") for i in range(3)
        ]

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=50),
            ),
        ):
            result = await manager._apply_session_observation_cap(  # pyright: ignore[reportPrivateUsage]
                observations, "session"
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_cap_disabled_skips_count_query(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A cap of 0 disables the limit and avoids the count query entirely."""
        monkeypatch.setattr(settings.DERIVER, "MAX_OBSERVATIONS_PER_SESSION", 0)
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )
        observations: list[ExplicitObservation | DeductiveObservation] = [
            self._explicit(f"obs {i}") for i in range(5)
        ]
        count_mock = AsyncMock(return_value=999)

        with patch(
            "src.crud.representation.crud.count_documents_for_session",
            new=count_mock,
        ):
            result = await manager._apply_session_observation_cap(  # pyright: ignore[reportPrivateUsage]
                observations, "session"
            )

        assert len(result) == 5
        count_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_representation_embeds_only_uncapped_observations(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """save_representation embeds and saves only the observations that fit the cap."""
        monkeypatch.setattr(settings.DERIVER, "MAX_OBSERVATIONS_PER_SESSION", 50)
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )
        representation = Representation(
            explicit=[self._explicit("a"), self._explicit("b"), self._explicit("c")]
        )

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=49),
            ),
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
                message_level_configuration=_resolved_config(),
            )

        # headroom = 50 - 49 = 1, so only the first observation survives.
        assert saved == 1
        mock_embed.assert_awaited_once_with(["a"])
        saved_observations = _saved_observations(mock_save)
        assert len(saved_observations) == 1
        assert saved_observations[0].content == "a"

    @pytest.mark.asyncio
    async def test_save_representation_skips_when_session_full(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """save_representation returns early without embedding when the session is full."""
        monkeypatch.setattr(settings.DERIVER, "MAX_OBSERVATIONS_PER_SESSION", 50)
        manager = RepresentationManager(
            "workspace", observer="observer", observed="observed"
        )
        representation = Representation(
            explicit=[self._explicit("a"), self._explicit("b")]
        )

        with (
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch(
                "src.crud.representation.crud.count_documents_for_session",
                new=AsyncMock(return_value=50),
            ),
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

        assert saved == 0
        mock_embed.assert_not_awaited()
        mock_save.assert_not_awaited()
