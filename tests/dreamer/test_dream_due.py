"""Tests for the read-only count of collections whose next dream is due."""

import datetime
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dreamer.dream_due import count_due_dreams
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


async def _make_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    internal_metadata: dict[str, object] | None = None,
) -> models.Collection:
    workspace, peer = sample_data
    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata=internal_metadata or {},
    )
    db_session.add(collection)
    await db_session.commit()
    return collection


async def _make_session(
    db_session: AsyncSession,
    workspace_name: str,
    configuration: dict[str, object] | None = None,
) -> str:
    session = models.Session(
        name=f"s-{generate_nanoid()}",
        workspace_name=workspace_name,
        configuration=configuration or {},
    )
    db_session.add(session)
    await db_session.commit()
    return session.name


async def _insert_docs(
    db_session: AsyncSession,
    collection: models.Collection,
    level: str,
    count: int,
    *,
    age_minutes: int = 0,
    session_name: str | None = None,
    sessionless: bool = False,
) -> None:
    if session_name is None and not sessionless:
        session_name = await _make_session(db_session, collection.workspace_name)
    created_at = _now() - datetime.timedelta(minutes=age_minutes)
    for _ in range(count):
        db_session.add(
            models.Document(
                content="test",
                level=level,
                workspace_name=collection.workspace_name,
                observer=collection.observer,
                observed=collection.observed,
                session_name=session_name,
                created_at=created_at,
            )
        )
    await db_session.commit()


async def _insert_dream_item(
    db_session: AsyncSession,
    collection: models.Collection,
    *,
    age_minutes: int,
    processed: bool,
    error: str | None = None,
) -> None:
    work_unit_key = construct_work_unit_key(
        collection.workspace_name,
        {
            "task_type": "dream",
            "observer": collection.observer,
            "observed": collection.observed,
            "dream_type": DreamType.OMNI.value,
        },
    )
    db_session.add(
        models.QueueItem(
            work_unit_key=work_unit_key,
            payload={"task_type": "dream"},
            task_type="dream",
            workspace_name=collection.workspace_name,
            processed=processed,
            error=error,
            created_at=_now() - datetime.timedelta(minutes=age_minutes),
        )
    )
    await db_session.commit()


@pytest.fixture(autouse=True)
def _pin_dream_config():  # pyright: ignore[reportUnusedFunction]
    with (
        patch("src.dreamer.dream_due.settings.DREAM.ENABLED", True),
        patch("src.dreamer.dream_due.settings.DREAM.DOCUMENT_THRESHOLD", 50),
        patch("src.dreamer.dream_due.settings.DREAM.ENABLED_TYPES", ["omni"]),
        patch("src.dreamer.dream_due.settings.DREAM.IDLE_TIMEOUT_MINUTES", 60),
        patch("src.dreamer.dream_due.settings.DREAM.MIN_HOURS_BETWEEN_DREAMS", 8),
    ):
        yield


@pytest.mark.asyncio
class TestCountDueDreams:
    async def test_below_threshold_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_derived_levels_do_not_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)
        await _insert_docs(db_session, collection, "deductive", 40, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_threshold_met_but_not_idle_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A collection still receiving documents is not idle yet."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=1)

        assert await count_due_dreams(db_session) == 0

    async def test_threshold_met_and_idle_is_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 1

    async def test_documents_since_last_dream_uses_stored_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_document_count": 40}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_min_hours_gate_blocks_a_recent_dream(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        last_dream_at = (_now() - datetime.timedelta(hours=2)).isoformat()
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_at": last_dream_at}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_naive_last_dream_at_is_read_as_utc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A stored timestamp with no offset must gate, not raise."""
        naive = (_now() - datetime.timedelta(hours=2)).replace(tzinfo=None).isoformat()
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_at": naive}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_pending_dream_item_blocks(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=10, processed=False
        )

        assert await count_due_dreams(db_session) == 0

    async def test_failed_dream_waits_for_new_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Without this the count never returns to zero."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=80, processed=True, error="boom"
        )

        assert await count_due_dreams(db_session) == 0

    async def test_failed_dream_retries_after_new_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=80, processed=True, error="boom"
        )
        await _insert_docs(db_session, collection, "explicit", 1, age_minutes=70)

        assert await count_due_dreams(db_session) == 1

    async def test_sessionless_documents_are_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """The deriver's own enqueue path refuses these, so they must not count."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(
            db_session, collection, "explicit", 60, age_minutes=90, sessionless=True
        )

        assert await count_due_dreams(db_session) == 0

    async def test_newest_document_decides_the_session(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=120)

        assert await count_due_dreams(db_session) == 1

        await _insert_docs(
            db_session, collection, "explicit", 1, age_minutes=90, sessionless=True
        )

        assert await count_due_dreams(db_session) == 0

    async def test_session_with_dreams_disabled_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A dream the enqueue path would refuse must not be counted."""
        collection = await _make_collection(db_session, sample_data)
        session_name = await _make_session(
            db_session,
            collection.workspace_name,
            {"dream": {"enabled": False}},
        )
        await _insert_docs(
            db_session,
            collection,
            "explicit",
            60,
            age_minutes=90,
            session_name=session_name,
        )

        assert await count_due_dreams(db_session) == 0

    async def test_dreams_disabled_globally_returns_zero(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch("src.dreamer.dream_due.settings.DREAM.ENABLED", False):
            assert await count_due_dreams(db_session) == 0

    async def test_card_refresh_is_never_counted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch(
            "src.dreamer.dream_due.settings.DREAM.ENABLED_TYPES", ["card_refresh"]
        ):
            assert await count_due_dreams(db_session) == 0
