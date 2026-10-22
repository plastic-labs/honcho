"""An in-flight claim must survive a work unit that runs past the stale timeout.

A claim's ``last_updated`` is otherwise bumped only when a queue ITEM completes, so
a task type whose whole unit is one long item -- the reconciler cycle, a dream --
never refreshes it and ``cleanup_stale_work_units`` reaps it mid-flight. Another
replica then claims the same unit and runs it concurrently; ``SKIP LOCKED`` stops
double-writing, not double-scanning.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.deriver import claim_heartbeat_interval_seconds, cleanup_stale_work_units
from src.deriver.queue_manager import QueueManager

WORK_UNIT_KEY = "representation:ws-hb:obs:obsd"

_AGE = text(
    "UPDATE active_queue_sessions SET last_updated = now() -"
    + " make_interval(mins => :mins) WHERE id = :id"
)
_EXISTS = text("SELECT count(*) FROM active_queue_sessions WHERE id = :id")
_READ = text("SELECT last_updated FROM active_queue_sessions WHERE id = :id")


async def _stale_claim(db: AsyncSession) -> str:
    """Insert a claim already old enough for the reaper, and return its id."""
    row = models.ActiveQueueSession(work_unit_key=WORK_UNIT_KEY)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    # Age it in SQL: last_updated carries an onupdate default, so an ORM write
    # of an explicit value does not survive the flush.
    await db.execute(
        _AGE, {"id": row.id, "mins": settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES + 1}
    )
    await db.commit()
    return row.id


def test_heartbeat_fires_well_inside_the_stale_window() -> None:
    """The two halves of the invariant must not be able to drift apart."""
    window = settings.DERIVER.STALE_SESSION_TIMEOUT_MINUTES * 60
    assert claim_heartbeat_interval_seconds() < window / 2, (
        "a heartbeat at or near the stale window cannot keep a claim alive"
    )


@pytest.mark.asyncio
async def test_a_heartbeat_saves_a_claim_from_the_reaper(
    db_session: AsyncSession,
) -> None:
    """The property that matters: a claim old enough to be reaped survives a beat."""
    aqs_id = await _stale_claim(db_session)
    before = (await db_session.execute(_READ, {"id": aqs_id})).scalar()

    assert await QueueManager()._touch_claim(WORK_UNIT_KEY, aqs_id) is True  # pyright: ignore[reportPrivateUsage]

    after = (await db_session.execute(_READ, {"id": aqs_id})).scalar()
    assert after is not None and before is not None and after > before, (
        "the heartbeat did not move last_updated"
    )

    await cleanup_stale_work_units(db_session)
    await db_session.commit()

    assert (await db_session.execute(_EXISTS, {"id": aqs_id})).scalar() == 1, (
        "heartbeat did not save the claim from the reaper"
    )


@pytest.mark.asyncio
async def test_without_a_heartbeat_the_same_claim_is_reaped(
    db_session: AsyncSession,
) -> None:
    """Counterpart, so the test above cannot pass vacuously."""
    aqs_id = await _stale_claim(db_session)

    await cleanup_stale_work_units(db_session)
    await db_session.commit()

    assert (await db_session.execute(_EXISTS, {"id": aqs_id})).scalar() == 0


@pytest.mark.asyncio
async def test_touch_claim_reports_an_already_reaped_claim(
    db_session: AsyncSession,
) -> None:
    """A beat matching no row means another worker may now own the unit."""
    aqs_id = await _stale_claim(db_session)
    await db_session.execute(
        text("DELETE FROM active_queue_sessions WHERE id = :id"), {"id": aqs_id}
    )
    await db_session.commit()

    assert await QueueManager()._touch_claim(WORK_UNIT_KEY, aqs_id) is False  # pyright: ignore[reportPrivateUsage]
