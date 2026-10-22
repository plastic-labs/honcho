"""AdvisoryLease: one holder at a time, released on release or on a dropped connection."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src import scheduler_lease
from src.scheduler_lease import AdvisoryLease


@pytest.fixture
def lease_name(db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(scheduler_lease, "service_engine", db_engine)
    return f"honcho:test-lease:{generate_nanoid()}"


@pytest.mark.asyncio
async def test_one_holder_at_a_time(lease_name: str) -> None:
    first, second = AdvisoryLease(lease_name), AdvisoryLease(lease_name)
    try:
        assert await first.ensure()
        assert await first.ensure()
        assert not await second.ensure()
        assert not second.held

        await first.release()

        assert await second.ensure()
    finally:
        await first.release()
        await second.release()


@pytest.mark.asyncio
async def test_a_dropped_holder_steps_down_and_another_takes_over(
    lease_name: str, db_engine: AsyncEngine
) -> None:
    first, second = AdvisoryLease(lease_name), AdvisoryLease(lease_name)
    try:
        assert await first.ensure()
        holder = first._connection  # pyright: ignore[reportPrivateUsage]
        assert holder is not None
        pid = await holder.scalar(text("SELECT pg_backend_pid()"))

        async with db_engine.connect() as admin:
            await admin.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})

        assert not await first.ensure()
        assert not first.held
        assert await second.ensure()
    finally:
        await first.release()
        await second.release()


@pytest.mark.asyncio
async def test_the_held_connection_is_not_in_a_transaction(
    lease_name: str, db_engine: AsyncEngine
) -> None:
    """An idle-in-transaction holder would pin a snapshot for as long as it leads."""
    lease = AdvisoryLease(lease_name)
    try:
        assert await lease.ensure()
        holder = lease._connection  # pyright: ignore[reportPrivateUsage]
        assert holder is not None
        pid = await holder.scalar(text("SELECT pg_backend_pid()"))

        async with db_engine.connect() as admin:
            state = await admin.scalar(
                text("SELECT state FROM pg_stat_activity WHERE pid = :pid"),
                {"pid": pid},
            )
        assert state == "idle"
    finally:
        await lease.release()
