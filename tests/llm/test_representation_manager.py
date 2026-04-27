from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.crud.representation import RepresentationManager
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)


@asynccontextmanager
async def _fake_tracked_db(_name: str):
    yield object()


class TestRepresentationManagerSave:
    @pytest.mark.asyncio
    async def test_save_representation_filters_blank_observations_before_embedding(self):
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
                message_level_configuration=SimpleNamespace(
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 1
        mock_embed.assert_awaited_once_with(["useful observation"])
        saved_observations = mock_save.await_args.args[1]
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
                message_level_configuration=SimpleNamespace(
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 1
        mock_embed.assert_awaited_once_with(["inferred conclusion"])
        saved_observations = mock_save.await_args.args[1]
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
                message_level_configuration=SimpleNamespace(
                    dream=SimpleNamespace(enabled=False)
                ),
            )

        assert saved == 0
        mock_embed.assert_not_awaited()
        mock_save.assert_not_awaited()
