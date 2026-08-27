"""Tests for the API-side dream due-check and for enqueueing a due dream."""

import datetime
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.backlog import find_due_dreams
from src.dreamer.dream_scheduler import execute_dream
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


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


async def _insert_docs(
    db_session: AsyncSession,
    collection: models.Collection,
    level: str,
    count: int,
    *,
    age_minutes: int = 0,
    session_name: str | None = None,
) -> None:
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
def _pin_dream_config():
    with (
        patch("src.backlog.settings.DREAM.ENABLED", True),
        patch("src.backlog.settings.DREAM.DOCUMENT_THRESHOLD", 50),
        patch("src.backlog.settings.DREAM.ENABLED_TYPES", ["omni"]),
        patch("src.backlog.settings.DREAM.IDLE_TIMEOUT_MINUTES", 60),
        patch("src.backlog.settings.DREAM.MIN_HOURS_BETWEEN_DREAMS", 8),
    ):
        yield


@pytest.mark.asyncio
class TestFindDueDreams:
    async def test_below_threshold_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)

        assert await find_due_dreams(db_session) == []

    async def test_derived_levels_do_not_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)
        await _insert_docs(db_session, collection, "deductive", 40, age_minutes=90)
        await _insert_docs(db_session, collection, "contradiction", 40, age_minutes=90)

        assert await find_due_dreams(db_session) == []

    async def test_threshold_met_but_not_idle_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=1)

        assert await find_due_dreams(db_session) == []

    async def test_threshold_met_and_idle_is_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        due = await find_due_dreams(db_session)

        assert len(due) == 1
        assert due[0].observer == collection.observer
        assert due[0].observed == collection.observed
        assert due[0].dream_type is DreamType.OMNI
        assert due[0].documents_since_last_dream == 60

    async def test_documents_since_last_dream_uses_stored_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_document_count": 40}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await find_due_dreams(db_session) == []

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

        assert await find_due_dreams(db_session) == []

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

        assert await find_due_dreams(db_session) == []

    async def test_failed_dream_waits_for_new_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=80, processed=True, error="boom"
        )

        assert await find_due_dreams(db_session) == []

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

        due = await find_due_dreams(db_session)

        assert len(due) == 1
        assert due[0].documents_since_last_dream == 61

    async def test_dreams_disabled_returns_nothing(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch("src.backlog.settings.DREAM.ENABLED", False):
            assert await find_due_dreams(db_session) == []

    async def test_card_refresh_is_never_enqueued(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch("src.backlog.settings.DREAM.ENABLED_TYPES", ["card_refresh"]):
            assert await find_due_dreams(db_session) == []


@pytest.mark.asyncio
class TestExecuteDream:
    async def test_enqueues_the_dream(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = models.Session(
            name=f"dream-session-{peer.name}", workspace_name=workspace.name
        )
        db_session.add(session)
        await db_session.commit()

        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(
            db_session,
            collection,
            "explicit",
            1,
            age_minutes=90,
            session_name=session.name,
        )

        with patch(
            "src.deriver.enqueue.enqueue_dream", new_callable=AsyncMock
        ) as mock_enqueue:
            await execute_dream(
                workspace.name,
                DreamType.OMNI,
                observer=peer.name,
                observed=peer.name,
                trigger_reason="document_threshold",
                delay_reason="idle_timeout",
            )

        assert mock_enqueue.called
        assert mock_enqueue.call_args.kwargs["session_name"] == session.name

    async def test_skips_when_no_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        await _make_collection(db_session, sample_data)

        with patch(
            "src.deriver.enqueue.enqueue_dream", new_callable=AsyncMock
        ) as mock_enqueue:
            await execute_dream(
                workspace.name,
                DreamType.OMNI,
                observer=peer.name,
                observed=peer.name,
            )

        assert not mock_enqueue.called


@pytest.mark.asyncio
class TestPollerEnqueueFailures:
    async def test_one_failing_dream_does_not_block_the_others(
        self,
        db_session: AsyncSession,  # pyright: ignore[reportUnusedParameter]
    ):
        from src.backlog import BacklogMetricsPoller, DueDream

        due = [
            DueDream(
                workspace_name="ws",
                observer=name,
                observed=name,
                dream_type=DreamType.OMNI,
                documents_since_last_dream=60,
            )
            for name in ("first", "second", "third")
        ]
        attempted: list[str] = []

        async def flaky(
            _workspace_name: str,
            _dream_type: DreamType,
            *,
            observer: str,
            observed: str,  # pyright: ignore[reportUnusedParameter]
            **_kwargs: object,
        ) -> None:
            attempted.append(observer)
            if observer == "first":
                raise ValueError("boom")

        with (
            patch("src.backlog.find_due_dreams", new=AsyncMock(return_value=due)),
            patch("src.backlog.execute_dream", new=AsyncMock(side_effect=flaky)),
        ):
            await BacklogMetricsPoller()._refresh()  # pyright: ignore[reportPrivateUsage]

        assert attempted == ["first", "second", "third"]
