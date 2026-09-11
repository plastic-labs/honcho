"""Interleavings that made the backlog aggregate disagree with the queue behind it.

Two writers on one work unit are the only way the aggregate can drift. The
enqueue path increments the backlog row it holds a lock on; a completion
recomputes that same row from the queue. Under READ COMMITTED a recompute that
aggregates *before* it takes the row lock reads a snapshot without the
concurrent enqueue in it, and then either clobbers the increment or — worse —
its membership DELETE waits on the enqueue's lock and drops the row anyway once
that enqueue commits, because the recheck re-evaluates the row rather than the
NOT EXISTS subquery. A live unit would then have no backlog row at all, and the
claim would never offer it again.

Each interleaving below was reproduced against the unfixed triggers. The
recompute now locks the backlog row first, so every statement after the wait
runs on a snapshot that includes whatever the wait was for.

The enqueue's own bounded retry is pinned here too: a deadlock breaking a
lock-order inversion is the residual case canonical lock ordering cannot rule
out, and before the retry the victim's whole batch was swallowed by the
fire-and-forget catch-all and silently never derived.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import pytest
from sqlalchemy import insert, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import models
from src.config import settings
from src.deriver.enqueue import _insert_queue_records
from src.models import DEFAULT_TENANT_ID
from tests.deriver.conftest import (
    SeedWorkUnit,
    _independent_sessions,
    _read_backlog_row,
)

# How long the blocked statement is given to prove it is really blocked, and how
# long it is then given to finish once the block is lifted. Both bound a hang
# rather than time anything, so they are generous relative to the work involved.
LOCK_WAIT_OBSERVATION_SECONDS = 0.5
UNBLOCKED_COMPLETION_TIMEOUT_SECONDS = 15.0


async def _committed_message_priced_at(
    seed_work_unit: SeedWorkUnit, *, unit_name: str, token_count: int
) -> int:
    """The id of a committed message whose ``token_count`` is exactly ``token_count``.

    The racing transactions insert their queue rows directly, and the insert
    trigger prices a representation item by looking its message up — so the
    message has to be committed before the race opens. Hanging it off an
    already-processed unit keeps it out of the backlog entirely.
    """
    seeded_items = await seed_work_unit(
        f"representation:priced-message:{unit_name}",
        token_counts=(token_count,),
        processed=True,
    )
    message_id = seeded_items[0].message_id
    assert message_id is not None
    return message_id


async def _enqueue_pending_item(
    session: AsyncSession,
    *,
    work_unit_key: str,
    workspace_name: str,
    message_id: int,
) -> None:
    """Insert one pending representation item, the shape the enqueue path writes.

    One statement, so the row-level insert trigger's backlog upsert — and the
    row lock it leaves behind until commit — is the whole of what this
    transaction holds.
    """
    await session.execute(
        insert(models.QueueItem).values(
            work_unit_key=work_unit_key,
            task_type="representation",
            payload={},
            processed=False,
            workspace_name=workspace_name,
            message_id=message_id,
            tenant_id=DEFAULT_TENANT_ID,
        )
    )


async def _mark_item_processed(session: AsyncSession, queue_item_id: int) -> None:
    """Complete one queue item, firing the statement-level recompute for its unit."""
    await session.execute(
        update(models.QueueItem)
        .where(models.QueueItem.id == queue_item_id)
        .values(processed=True)
    )


@pytest.mark.asyncio
class TestACompletionRacingAnEnqueue:
    """A recompute that overlaps an enqueue has to end up counting both writers."""

    async def test_the_delete_guard_keeps_a_unit_whose_next_item_arrived_mid_recompute(
        self,
        db_engine: AsyncEngine,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """Draining a unit's last item while its next one is in flight must not delete the row.

        The enqueue holds the backlog row's lock uncommitted, so the completion's
        recompute blocks. Aggregating before that wait — the unfixed order — left
        the recompute holding an empty result and falling through to the
        membership DELETE, whose row-level recheck cannot see that the NOT EXISTS
        subquery has stopped being true. The unit lost its backlog row while it
        still had pending work, and nothing recreates it: the claim skips what has
        no row, so the item sat unprocessed until an unrelated enqueue for the
        same unit happened to recreate it.
        """
        workspace, _peer = sample_data
        work_unit_key = "representation:delete-guard-race"
        seeded_items = await seed_work_unit(work_unit_key, token_counts=(5,))
        enqueued_message_id = await _committed_message_priced_at(
            seed_work_unit, unit_name="delete-guard-race", token_count=41
        )

        sessions = _independent_sessions(db_engine)
        async with sessions() as enqueuing, sessions() as completing:
            await _enqueue_pending_item(
                enqueuing,
                work_unit_key=work_unit_key,
                workspace_name=workspace.name,
                message_id=enqueued_message_id,
            )
            completion = asyncio.create_task(
                _mark_item_processed(completing, seeded_items[0].id)
            )
            try:
                await asyncio.sleep(LOCK_WAIT_OBSERVATION_SECONDS)

                assert not completion.done(), (
                    "the recompute must block on the enqueue's backlog row lock — "
                    + "finishing first means it aggregated a snapshot without the "
                    + "enqueued item in it"
                )

                await enqueuing.commit()
                await asyncio.wait_for(completion, UNBLOCKED_COMPLETION_TIMEOUT_SECONDS)
            finally:
                completion.cancel()

            recomputed_row = await _read_backlog_row(completing, work_unit_key)
            await completing.commit()

        committed_row = await _read_backlog_row(db_session, work_unit_key)

        assert recomputed_row is not None, (
            "the recompute saw the enqueued item once it woke, so the unit is "
            + "still pending work and keeps its row"
        )
        assert recomputed_row.pending_count == 1

        assert committed_row is not None, (
            "the unit has an unprocessed item, so it must still be in the backlog"
        )
        assert committed_row.pending_count == 1
        assert committed_row.total_tokens == 41

    async def test_the_recompute_counts_the_enqueued_item_instead_of_clobbering_it(
        self,
        db_engine: AsyncEngine,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        seed_work_unit: SeedWorkUnit,
    ) -> None:
        """A recompute over a still-live unit must include the item that landed while it waited.

        Here the unit survives either way — two items remain pending — so the
        damage is silent rather than fatal: aggregating before the lock produced
        a row counting one item and pricing only the seeded one, and the batch
        gate then held the unit back until the age flush because its token total
        was short by the whole enqueued item.
        """
        workspace, _peer = sample_data
        work_unit_key = "representation:clobber-race"
        seeded_items = await seed_work_unit(
            work_unit_key, token_counts=(7, 13), ages_seconds=(60, 30)
        )
        enqueued_message_id = await _committed_message_priced_at(
            seed_work_unit, unit_name="clobber-race", token_count=101
        )

        sessions = _independent_sessions(db_engine)
        async with sessions() as enqueuing, sessions() as completing:
            await _enqueue_pending_item(
                enqueuing,
                work_unit_key=work_unit_key,
                workspace_name=workspace.name,
                message_id=enqueued_message_id,
            )
            completion = asyncio.create_task(
                _mark_item_processed(completing, seeded_items[0].id)
            )
            try:
                await asyncio.sleep(LOCK_WAIT_OBSERVATION_SECONDS)

                assert not completion.done(), (
                    "the recompute must block on the enqueue's backlog row lock"
                )

                await enqueuing.commit()
                await asyncio.wait_for(completion, UNBLOCKED_COMPLETION_TIMEOUT_SECONDS)
            finally:
                completion.cancel()

            await completing.commit()

        committed_row = await _read_backlog_row(db_session, work_unit_key)

        assert committed_row is not None
        assert committed_row.pending_count == 2, (
            "the surviving seeded item and the enqueued one, not the seeded one alone"
        )
        assert committed_row.total_tokens == 114, (
            "13 from the item that stayed pending plus 101 from the enqueued one"
        )


@pytest.fixture
def single_tenant_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unprefixed work unit keys stamp a NULL tenant, which is only legal flag-off."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)


class _DeadlockDetected(Exception):
    """Stands in for the driver error raised at the losing end of a deadlock."""

    sqlstate: str = "40P01"


def _deadlock_error() -> DBAPIError:
    """A DBAPIError shaped the way the retry classifier sniffs one: SQLSTATE on ``orig``."""
    return DBAPIError(
        "INSERT INTO queue ...", None, _DeadlockDetected("deadlock detected")
    )


def _queue_records(*work_unit_keys: str) -> list[dict[str, Any]]:
    """Minimal enqueue records — only the fields the insert path itself reads."""
    return [
        {
            "work_unit_key": work_unit_key,
            "task_type": "representation",
            "payload": {},
            "session_id": None,
            "workspace_name": "workspace",
            "message_id": None,
        }
        for work_unit_key in work_unit_keys
    ]


class _RecordingEnqueueSession:
    """A stand-in for the enqueue's session that records its batches and fails on cue.

    The deadlock this covers is a genuine race between the insert trigger's lock
    order and the claim's, which no seeding can force deterministically. Driving
    the retry from the session boundary instead pins what the retry actually
    promises — one more attempt, then a commit — without a flaky reproduction.
    """

    def __init__(self, *, failures: Sequence[BaseException] = ()) -> None:
        self.executed_batches: list[list[dict[str, Any]]] = []
        self.commit_count: int = 0
        self.rollback_count: int = 0
        self._failures: list[BaseException] = list(failures)

    async def execute(self, _statement: Any, parameters: Any = None) -> None:
        self.executed_batches.append(list(parameters or []))
        if self._failures:
            raise self._failures.pop(0)

    async def commit(self) -> None:
        self.commit_count += 1

    async def rollback(self) -> None:
        self.rollback_count += 1


def _as_session(recording_session: _RecordingEnqueueSession) -> AsyncSession:
    """Hand the double to the code under test.

    ``_insert_queue_records`` only ever calls execute / commit / rollback, so the
    double covers the whole of the surface it uses; the cast goes through
    ``object`` because a partial stand-in deliberately does not structurally
    overlap the real session.
    """
    return cast(AsyncSession, cast(object, recording_session))


@pytest.mark.asyncio
class TestTheEnqueueRetriesADeadlockedBatch:
    """A deadlock victim retries; anything terminal is left to the caller."""

    async def test_a_deadlocked_batch_is_retried_and_then_commits(
        self,
        single_tenant_enqueue: None,  # noqa: ARG002  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """The whole point: the batch reaches the queue instead of vanishing."""
        recording_session = _RecordingEnqueueSession(failures=[_deadlock_error()])

        await _insert_queue_records(
            _as_session(recording_session),
            _queue_records(
                "representation:workspace:beta", "representation:workspace:alpha"
            ),
        )

        assert len(recording_session.executed_batches) == 2, (
            "the deadlocked attempt plus exactly one retry"
        )
        assert recording_session.commit_count == 1
        assert recording_session.rollback_count == 1, (
            "the aborted transaction is rolled back before the retry reuses the session"
        )

    async def test_a_terminal_error_propagates_without_a_retry(
        self,
        single_tenant_enqueue: None,  # noqa: ARG002  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """Retrying what cannot self-heal only delays the report of a real bug."""
        recording_session = _RecordingEnqueueSession(
            failures=[ValueError("bad record")]
        )

        with pytest.raises(ValueError, match="bad record"):
            await _insert_queue_records(
                _as_session(recording_session),
                _queue_records("representation:workspace:alpha"),
            )

        assert len(recording_session.executed_batches) == 1
        assert recording_session.commit_count == 0
        assert recording_session.rollback_count == 1

    async def test_the_batch_reaches_the_insert_sorted_by_work_unit_key(
        self,
        single_tenant_enqueue: None,  # noqa: ARG002  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """Canonical key order is what keeps the insert trigger's locks free of inversions."""
        unsorted_keys = (
            "representation:workspace:gamma",
            "representation:workspace:alpha",
            "representation:workspace:beta",
        )
        recording_session = _RecordingEnqueueSession()

        await _insert_queue_records(
            _as_session(recording_session), _queue_records(*unsorted_keys)
        )

        assert [
            record["work_unit_key"] for record in recording_session.executed_batches[0]
        ] == sorted(unsorted_keys)
