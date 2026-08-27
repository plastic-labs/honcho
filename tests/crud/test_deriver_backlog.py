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
        created_at=datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(seconds=age_seconds),
    )
    db.add(item)
    await db.flush()
    return item


class TestDeriverBacklog:
    async def test_empty_queue_reports_zero(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.eligible_work_units == 0
        assert backlog.pending_items == 0
        assert backlog.oldest_pending_age_seconds == 0.0

    async def test_sub_threshold_batch_is_pending_but_not_eligible(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A small, fresh batch is real work that a deriver would not yet claim.

        This is the case that makes eligible and pending different numbers, and
        the reason pending exists as its own gauge.
        """
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

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.pending_items == 1
        assert backlog.eligible_work_units == 0

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

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.eligible_work_units == 1

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

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.eligible_work_units == 1
        assert backlog.oldest_pending_age_seconds >= (
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

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.eligible_work_units == 1

    async def test_live_claim_hides_work_from_the_count(
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
            work_unit_key="representation:claimed",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        db_session.add(
            models.ActiveQueueSession(work_unit_key="representation:claimed")
        )
        await db_session.commit()

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.eligible_work_units == 0

    async def test_stale_claim_is_reaped_and_stops_hiding_work(
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
            work_unit_key="representation:abandoned",
            token_count=settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        )
        stale_cutoff = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(
            minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES + 1
        )
        db_session.add(
            models.ActiveQueueSession(
                work_unit_key="representation:abandoned",
                last_updated=stale_cutoff,
            )
        )
        await db_session.commit()

        assert (await crud.get_deriver_backlog(db_session)).eligible_work_units == 0

        await crud.cleanup_stale_work_units()
        await db_session.commit()

        assert (await crud.get_deriver_backlog(db_session)).eligible_work_units == 1

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

        backlog = await crud.get_deriver_backlog(db_session)

        assert backlog.pending_items == 0
        assert backlog.eligible_work_units == 0
        assert backlog.oldest_pending_age_seconds == 0.0


class TestBacklogAgreesWithDeriver:
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

        expected = (await crud.get_deriver_backlog(db_session)).eligible_work_units
        claimed = await QueueManager().get_and_claim_work_units()

        assert len(claimed) == expected
