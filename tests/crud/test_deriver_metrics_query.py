import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings

pytestmark = pytest.mark.asyncio


async def _make_session(
    db: AsyncSession, workspace: models.Workspace
) -> models.Session:
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add(session)
    await db.flush()
    return session


async def _add_representation_item(
    db: AsyncSession,
    workspace: models.Workspace,
    peer: models.Peer,
    session: models.Session,
    *,
    work_unit_key: str,
    token_count: int,
    age_seconds: int = 0,
    seq: int = 1,
) -> models.QueueItem:
    message = models.Message(
        session_name=session.name,
        content="x",
        token_count=token_count,
        seq_in_session=seq,
        peer_name=peer.name,
        workspace_name=workspace.name,
    )
    db.add(message)
    await db.flush()

    item = models.QueueItem(
        session_id=session.id,
        work_unit_key=work_unit_key,
        task_type="representation",
        payload={},
        processed=False,
        workspace_name=workspace.name,
        message_id=message.id,
        created_at=datetime.datetime.now(datetime.UTC)
        - datetime.timedelta(seconds=age_seconds),
    )
    db.add(item)
    await db.flush()
    return item


async def _add_message(
    db: AsyncSession,
    workspace: models.Workspace,
    peer: models.Peer,
    session: models.Session,
    *,
    seq: int = 1,
) -> models.Message:
    message = models.Message(
        session_name=session.name,
        content="x",
        token_count=1,
        seq_in_session=seq,
        peer_name=peer.name,
        workspace_name=workspace.name,
    )
    db.add(message)
    await db.flush()
    return message


def _stale_timestamp() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES + 1
    )


class TestDeriverMetrics:
    async def test_empty_queue_reports_zero(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],  # pyright: ignore[reportUnusedParameter]
    ):
        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 0
        assert stats.claimed_work_units == 0
        assert stats.pending_items == 0
        assert stats.oldest_pending_age_seconds == 0.0

    async def test_sub_threshold_batch_is_pending_but_not_eligible(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A small, fresh batch is real work that a deriver would not yet claim."""
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:small",
            token_count=1,
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.pending_items == 1
        assert stats.eligible_work_units == 0

    async def test_token_threshold_makes_batch_eligible(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:big",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 1

    async def test_age_flush_makes_sub_threshold_batch_eligible(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:old",
            token_count=1,
            age_seconds=settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS + 60,
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 1
        assert stats.oldest_pending_age_seconds >= (
            settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS
        )

    async def test_non_representation_work_is_eligible_immediately(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, _peer = sample_data

        db_session.add(
            models.QueueItem(
                work_unit_key="reconciler:sync_vectors",
                task_type="reconciler",
                payload={},
                processed=False,
                workspace_name=workspace.name,
            )
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 1

    async def test_live_claim_is_counted_as_work_in_flight(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A claimed work unit is not claimable, but it is still outstanding work."""
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:claimed",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        db_session.add(
            models.ActiveQueueSession(work_unit_key="representation:claimed")
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 0
        assert stats.claimed_work_units == 1

    async def test_stale_claim_does_not_hide_work_and_is_not_in_flight(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A dead worker's claim must not read as in flight, and must not hide work."""
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:abandoned",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        db_session.add(
            models.ActiveQueueSession(
                work_unit_key="representation:abandoned",
                last_updated=_stale_timestamp(),
            )
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.eligible_work_units == 1
        assert stats.claimed_work_units == 0

    async def test_processed_items_are_not_counted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        item = await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:done",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        item.processed = True
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.pending_items == 0
        assert stats.eligible_work_units == 0
        assert stats.oldest_pending_age_seconds == 0.0


class TestPendingEmbeddings:
    async def test_never_attempted_row_is_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)
        message = await _add_message(db_session, workspace, peer, session)
        db_session.add(
            models.MessageEmbedding(
                content="x",
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="pending",
            )
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.embeddings_pending == 1
        assert stats.embeddings_pending_due == 1

    async def test_row_inside_its_retry_wait_is_pending_but_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A backing-off row is work the deriver cannot act on yet."""
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)
        message = await _add_message(db_session, workspace, peer, session)
        db_session.add(
            models.MessageEmbedding(
                content="x",
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="pending",
                last_sync_at=datetime.datetime.now(datetime.UTC),
                sync_attempts=1,
            )
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.embeddings_pending == 1
        assert stats.embeddings_pending_due == 0

    async def test_synced_rows_are_not_counted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)
        message = await _add_message(db_session, workspace, peer, session)
        db_session.add(
            models.MessageEmbedding(
                content="x",
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="synced",
            )
        )
        await db_session.commit()

        stats = await crud.get_deriver_metrics(db_session)

        assert stats.embeddings_pending == 0
        assert stats.embeddings_pending_due == 0


class TestMetricsAgreeWithDeriver:
    @pytest.mark.parametrize(
        "token_count,age_seconds",
        [
            (1, 0),
            (settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS, 0),
            (1, settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS + 60),
        ],
        ids=["sub-threshold", "token-threshold", "age-flush"],
    )
    async def test_eligible_count_matches_what_the_deriver_claims(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        token_count: int,
        age_seconds: int,
    ):
        """The gauge is only trustworthy if it uses the deriver's own rule."""
        from src.deriver.queue_manager import QueueManager

        workspace, peer = sample_data
        session = await _make_session(db_session, workspace)

        await _add_representation_item(
            db_session,
            workspace,
            peer,
            session,
            work_unit_key="representation:agreement",
            token_count=token_count,
            age_seconds=age_seconds,
        )
        await db_session.commit()

        expected = (await crud.get_deriver_metrics(db_session)).eligible_work_units
        claimed = await QueueManager().get_and_claim_work_units()

        assert len(claimed) == expected
