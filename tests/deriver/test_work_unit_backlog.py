"""Behavior of the trigger-maintained work_unit_backlog and the claim that reads it.

The backlog holds one row per work unit that has anything unprocessed in the
queue: enqueue increments it, completion and deletion recompute it from the
queue, and a unit with nothing pending has no row at all — existence, not
``pending_count``, is the signal. The deriver's claim reads that table instead
of re-aggregating the queue, so these tests pin both halves: the aggregate the
triggers maintain, and what the claim and the metrics query do with it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Select, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.config import settings
from src.crud.deriver import claim_rows_query
from src.deriver.queue_manager import QueueManager
from src.models import DEFAULT_TENANT_ID
from tests.deriver.conftest import (
    SeedWorkUnit,
    _independent_sessions,
    _read_backlog_keys,
    _read_backlog_row,
)


async def _count_unprocessed_queue_items(db: AsyncSession) -> int:
    """The queue's own answer for pending items, for parity with the aggregate."""
    result = await db.execute(
        select(func.count())
        .select_from(models.QueueItem)
        .where(~models.QueueItem.processed)
    )
    return int(result.scalar_one())


def _claim_candidate_query(limit: int) -> Select[Any]:
    """The real claim query, straight from the production builder — no mirrored
    SQL, so these tests exercise the exact shape the deriver runs and drift is
    impossible."""
    return claim_rows_query(limit)


@pytest.mark.asyncio
class TestEnqueueMaintainsTheBacklog:
    """The insert direction: one incremented row per work unit."""

    async def test_enqueued_items_aggregate_into_one_backlog_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """A unit's row carries its pending count, its token sum and its oldest item."""
        work_unit_key = "representation:aggregate"
        queue_items = await seed_work_unit(
            work_unit_key, token_counts=(5, 7, 11), ages_seconds=(30, 20, 10)
        )

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 3
        assert backlog_row.total_tokens == 23
        assert backlog_row.task_type == "representation"
        assert backlog_row.tenant_id == DEFAULT_TENANT_ID
        assert backlog_row.oldest_created_at == min(
            item.created_at for item in queue_items
        )

    async def test_a_later_item_with_an_older_timestamp_moves_oldest_back(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """The fast path keeps the least timestamp, not the most recently written one."""
        work_unit_key = "representation:least-timestamp"
        await seed_work_unit(work_unit_key, token_counts=(3,), ages_seconds=(10,))
        backdated_items = await seed_work_unit(
            work_unit_key, token_counts=(4,), ages_seconds=(600,)
        )

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 2
        assert backlog_row.total_tokens == 7
        assert backlog_row.oldest_created_at == backdated_items[0].created_at

    async def test_tenantless_queue_rows_still_sum_their_tokens(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """Without a tenant on the row the token probe falls back to the plain id."""
        work_unit_key = "representation:tenantless"
        await seed_work_unit(work_unit_key, token_counts=(6, 9), tenant_id=None)

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.tenant_id is None
        assert backlog_row.total_tokens == 15

    async def test_an_item_enqueued_processed_never_creates_a_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """Only unprocessed work belongs in the backlog."""
        work_unit_key = "representation:born-processed"
        await seed_work_unit(work_unit_key, token_counts=(50,), processed=True)

        assert await _read_backlog_row(db_session, work_unit_key) is None


@pytest.mark.asyncio
class TestCompletionRecomputesTheBacklog:
    """Marking items processed recomputes the row over what is left."""

    async def test_completing_the_oldest_item_shrinks_the_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """Count, tokens and oldest timestamp all follow the remaining items."""
        work_unit_key = "representation:partial-completion"
        queue_items = await seed_work_unit(
            work_unit_key, token_counts=(5, 7, 11), ages_seconds=(30, 20, 10)
        )

        queue_items[0].processed = True
        await db_session.commit()

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 2
        assert backlog_row.total_tokens == 18
        assert backlog_row.oldest_created_at == queue_items[1].created_at

    async def test_completing_every_item_removes_the_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """A drained unit leaves no row behind, not a zeroed one."""
        work_unit_key = "representation:full-completion"
        queue_items = await seed_work_unit(work_unit_key, token_counts=(5, 7))

        for queue_item in queue_items:
            queue_item.processed = True
        await db_session.commit()

        assert await _read_backlog_row(db_session, work_unit_key) is None
        assert await _read_backlog_keys(db_session) == set()

    async def test_reopening_a_completed_item_brings_the_row_back(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """A retry flips processed back, and the unit becomes pending work again."""
        work_unit_key = "representation:reopened"
        queue_items = await seed_work_unit(work_unit_key, token_counts=(9,))

        queue_items[0].processed = True
        await db_session.commit()
        queue_items[0].processed = False
        await db_session.commit()

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 1
        assert backlog_row.total_tokens == 9


@pytest.mark.asyncio
class TestDeletionRecomputesTheBacklog:
    """Deleting queue rows recomputes the row, and only when it was pending."""

    async def test_deleting_an_unprocessed_item_shrinks_the_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """A cancelled item leaves the row exact over what remains, not decremented."""
        work_unit_key = "representation:partial-delete"
        queue_items = await seed_work_unit(
            work_unit_key, token_counts=(5, 7, 11), ages_seconds=(30, 20, 10)
        )

        await db_session.execute(
            delete(models.QueueItem).where(models.QueueItem.id == queue_items[0].id)
        )
        await db_session.commit()

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 2
        assert backlog_row.total_tokens == 18
        assert backlog_row.oldest_created_at == queue_items[1].created_at

    async def test_deleting_the_last_unprocessed_item_removes_the_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """The guard keys on queue membership, so the row goes with the last item."""
        work_unit_key = "representation:full-delete"
        queue_items = await seed_work_unit(work_unit_key, token_counts=(5, 7))

        await db_session.execute(
            delete(models.QueueItem).where(
                models.QueueItem.id.in_([item.id for item in queue_items])
            )
        )
        await db_session.commit()

        assert await _read_backlog_row(db_session, work_unit_key) is None

    async def test_deleting_a_processed_item_leaves_the_row_alone(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """Cleanup of completed rows must not disturb a unit's live aggregate."""
        work_unit_key = "representation:delete-processed"
        pending_items = await seed_work_unit(
            work_unit_key, token_counts=(5, 7), ages_seconds=(30, 20)
        )
        completed_items = await seed_work_unit(
            work_unit_key, token_counts=(100,), processed=True
        )
        before = await _read_backlog_row(db_session, work_unit_key)

        await db_session.execute(
            delete(models.QueueItem).where(models.QueueItem.id == completed_items[0].id)
        )
        await db_session.commit()

        after = await _read_backlog_row(db_session, work_unit_key)

        assert before is not None
        assert after is not None
        assert after.pending_count == before.pending_count == 2
        assert after.total_tokens == before.total_tokens == 12
        assert after.oldest_created_at == pending_items[0].created_at


@pytest.mark.asyncio
class TestNonRepresentationUnits:
    """Infrastructure work carries no tokens and never waits on the batch gate."""

    async def test_a_webhook_unit_carries_no_tokens_and_is_claimable(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Webhook work has no message to price, and no token target to clear."""
        batch_gate_settings(target_tokens=10_000, workers=2)
        work_unit_key = "webhook:seeded-workspace"
        await seed_work_unit(
            work_unit_key,
            task_type="webhook",
            token_counts=(0,),
            with_messages=False,
        )

        backlog_row = await _read_backlog_row(db_session, work_unit_key)
        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert backlog_row is not None
        assert backlog_row.task_type == "webhook"
        assert backlog_row.total_tokens == 0
        assert set(claimed_work_units) == {work_unit_key}

    async def test_a_deletion_unit_ignores_the_tokens_of_its_message(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """The token lookup is typed by task_type, not by whether a message is attached."""
        batch_gate_settings(target_tokens=10_000, workers=2)
        work_unit_key = "deletion:seeded-workspace:session:abc"
        await seed_work_unit(work_unit_key, task_type="deletion", token_counts=(500,))

        backlog_row = await _read_backlog_row(db_session, work_unit_key)
        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert backlog_row is not None
        assert backlog_row.total_tokens == 0
        assert set(claimed_work_units) == {work_unit_key}


@pytest.mark.asyncio
class TestTriggerEffectsAreTransactional:
    """Trigger writes live and die with the transaction that caused them."""

    async def test_a_rolled_back_enqueue_leaves_no_backlog_row(
        self,
        db_engine: AsyncEngine,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """The increment is visible inside the transaction and gone once it aborts."""
        workspace, _peer = sample_data
        work_unit_key = "representation:rolled-back"

        async with _independent_sessions(db_engine)() as writer:
            writer.add(
                models.QueueItem(
                    work_unit_key=work_unit_key,
                    task_type="representation",
                    payload={},
                    processed=False,
                    workspace_name=workspace.name,
                    tenant_id=DEFAULT_TENANT_ID,
                )
            )
            await writer.flush()
            uncommitted_row = await _read_backlog_row(writer, work_unit_key)
            await writer.rollback()

        assert uncommitted_row is not None
        assert await _read_backlog_row(db_session, work_unit_key) is None

    async def test_a_deduped_insert_never_counts_toward_the_backlog(
        self,
        db_engine: AsyncEngine,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """The dedup loser hits the partial unique index and rolls its increment back."""
        work_unit_key = "reconciler:sync_vectors"
        await seed_work_unit(
            work_unit_key,
            task_type="reconciler",
            token_counts=(0,),
            tenant_id=None,
            with_messages=False,
        )

        async with _independent_sessions(db_engine)() as loser:
            loser.add(
                models.QueueItem(
                    work_unit_key=work_unit_key,
                    task_type="reconciler",
                    payload={},
                    processed=False,
                )
            )
            with pytest.raises(IntegrityError):
                await loser.flush()
            await loser.rollback()

        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert backlog_row is not None
        assert backlog_row.pending_count == 1


@pytest.mark.asyncio
class TestClaimReadsTheBacklog:
    """What the deriver claims is decided entirely by the aggregate rows."""

    async def test_eligible_units_are_claimed_oldest_first(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Successive claims walk the backlog in oldest-first order."""
        batch_gate_settings(target_tokens=10, workers=1)
        oldest_key = "representation:oldest"
        middle_key = "representation:middle"
        newest_key = "representation:newest"
        await seed_work_unit(oldest_key, token_counts=(50,), ages_seconds=(300,))
        await seed_work_unit(middle_key, token_counts=(50,), ages_seconds=(200,))
        await seed_work_unit(newest_key, token_counts=(50,), ages_seconds=(100,))

        queue_manager = QueueManager()
        first_claim = await queue_manager.get_and_claim_work_units()
        second_claim = await queue_manager.get_and_claim_work_units()
        third_claim = await queue_manager.get_and_claim_work_units()

        assert set(first_claim) == {oldest_key}
        assert set(second_claim) == {middle_key}
        assert set(third_claim) == {newest_key}

    async def test_a_unit_at_the_token_target_is_claimable_while_young(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Reaching the token target is the ordinary way a batch becomes claimable."""
        batch_gate_settings(target_tokens=512, max_age_seconds=1800, workers=2)
        work_unit_key = "representation:at-target"
        await seed_work_unit(work_unit_key, token_counts=(256, 256))

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert set(claimed_work_units) == {work_unit_key}

    async def test_a_young_sub_target_unit_is_pending_but_not_claimable(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """A small, fresh batch is real work the deriver deliberately leaves waiting."""
        batch_gate_settings(target_tokens=512, max_age_seconds=1800, workers=2)
        work_unit_key = "representation:accumulating"
        await seed_work_unit(work_unit_key, token_counts=(10,))

        claimed_work_units = await QueueManager().get_and_claim_work_units()
        backlog_row = await _read_backlog_row(db_session, work_unit_key)

        assert claimed_work_units == {}
        assert backlog_row is not None
        assert backlog_row.pending_count == 1

    async def test_a_sub_target_unit_becomes_claimable_once_it_ages_past_the_flush(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """The age flush reads oldest_created_at, so a backdated row claims immediately."""
        batch_gate_settings(target_tokens=512, max_age_seconds=60, workers=2)
        work_unit_key = "representation:age-flushed"
        await seed_work_unit(work_unit_key, token_counts=(10,), ages_seconds=(600,))

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert set(claimed_work_units) == {work_unit_key}

    async def test_a_claimed_unit_is_excluded_from_the_candidate_set(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """An active claim row hides its unit from the backlog SELECT."""
        batch_gate_settings(target_tokens=10, workers=5)
        claimed_key = "representation:already-claimed"
        free_key = "representation:still-free"
        await seed_work_unit(claimed_key, token_counts=(50,), ages_seconds=(300,))
        await seed_work_unit(free_key, token_counts=(50,), ages_seconds=(200,))
        db_session.add(models.ActiveQueueSession(work_unit_key=claimed_key))
        await db_session.commit()

        candidate_keys = (
            (await db_session.execute(_claim_candidate_query(limit=10))).scalars().all()
        )
        # Release the FOR UPDATE locks that mirrored SELECT took, or the real
        # claim below would skip the very row this test expects it to take.
        await db_session.rollback()
        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert list(candidate_keys) == [free_key]
        assert set(claimed_work_units) == {free_key}
        assert await _read_backlog_keys(db_session) == {claimed_key, free_key}

    async def test_a_stale_claim_row_still_hides_its_unit_from_the_claim(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Stale claims are released by the cleanup pass, never by the claim itself."""
        batch_gate_settings(target_tokens=10, workers=5)
        work_unit_key = "representation:stale-claim"
        await seed_work_unit(work_unit_key, token_counts=(50,))
        db_session.add(
            models.ActiveQueueSession(
                work_unit_key=work_unit_key,
                last_updated=datetime.now(UTC)
                - timedelta(minutes=settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES + 1),
            )
        )
        await db_session.commit()

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert claimed_work_units == {}

    async def test_a_drained_unit_disappears_from_the_candidate_set(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Completion is what makes a unit stop being offered, via its vanished row."""
        batch_gate_settings(target_tokens=10, workers=5)
        work_unit_key = "representation:drained"
        queue_items = await seed_work_unit(work_unit_key, token_counts=(50,))

        for queue_item in queue_items:
            queue_item.processed = True
        await db_session.commit()

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert claimed_work_units == {}
        assert await _read_backlog_keys(db_session) == set()


@pytest.mark.asyncio
class TestConcurrentClaimersTakeDisjointUnits:
    """SKIP LOCKED on the backlog rows is what keeps two derivers from colliding."""

    async def test_two_open_transactions_select_disjoint_candidate_sets(
        self,
        db_engine: AsyncEngine,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Both transactions stay open at once, so the second really is skipping locks."""
        batch_gate_settings(target_tokens=10, workers=2)
        keys_by_age = [
            "representation:concurrent-1",
            "representation:concurrent-2",
            "representation:concurrent-3",
            "representation:concurrent-4",
        ]
        for age_seconds, work_unit_key in zip(
            (400, 300, 200, 100), keys_by_age, strict=True
        ):
            await seed_work_unit(
                work_unit_key, token_counts=(50,), ages_seconds=(age_seconds,)
            )

        sessions = _independent_sessions(db_engine)
        candidate_query = _claim_candidate_query(limit=2)
        async with (
            sessions() as first_claimer,
            first_claimer.begin(),
            sessions() as second_claimer,
            second_claimer.begin(),
        ):
            first_keys = list(
                (await first_claimer.execute(candidate_query)).scalars().all()
            )
            second_keys = list(
                (await second_claimer.execute(candidate_query)).scalars().all()
            )

            assert first_keys == keys_by_age[:2]
            assert second_keys == keys_by_age[2:]
            assert set(first_keys).isdisjoint(second_keys)


@pytest.mark.asyncio
class TestMetricsReadTheSameBacklog:
    """The metrics query must agree with the queue it summarizes."""

    async def test_pending_items_matches_every_unprocessed_queue_row(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """Pending counts items, eligible and claimed count units."""
        batch_gate_settings(target_tokens=512, max_age_seconds=1800)
        eligible_key = "representation:metrics-eligible"
        waiting_key = "representation:metrics-waiting"
        claimed_key = "webhook:metrics-claimed"
        await seed_work_unit(
            eligible_key, token_counts=(300, 300), ages_seconds=(60, 30)
        )
        await seed_work_unit(waiting_key, token_counts=(10,))
        await seed_work_unit(
            claimed_key, task_type="webhook", token_counts=(0,), with_messages=False
        )
        await seed_work_unit(eligible_key, token_counts=(999,), processed=True)
        db_session.add(models.ActiveQueueSession(work_unit_key=claimed_key))
        await db_session.commit()

        metrics = await crud.get_deriver_metrics(db_session)

        assert metrics.pending_items == await _count_unprocessed_queue_items(db_session)
        assert metrics.pending_items == 4
        assert metrics.eligible_work_units == 1
        assert metrics.claimed_work_units == 1
        assert metrics.oldest_pending_age_seconds >= 60

    async def test_metrics_return_to_zero_once_the_queue_drains(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
    ) -> None:
        """With every backlog row gone the gauges read zero, not a stale total."""
        batch_gate_settings(target_tokens=10)
        work_unit_key = "representation:metrics-drained"
        queue_items = await seed_work_unit(work_unit_key, token_counts=(50, 50))

        for queue_item in queue_items:
            queue_item.processed = True
        await db_session.commit()

        metrics = await crud.get_deriver_metrics(db_session)

        assert metrics.pending_items == 0
        assert metrics.eligible_work_units == 0
        assert metrics.oldest_pending_age_seconds == 0.0
