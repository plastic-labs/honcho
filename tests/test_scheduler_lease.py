"""AdvisoryLease: one holder at a time, released on release or on a dropped connection."""

import asyncio
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src import scheduler_lease
from src.scheduler_lease import AdvisoryLease


@pytest.fixture
def lease_name(db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(
        scheduler_lease,
        "service_read_engine",
        db_engine.execution_options(isolation_level="AUTOCOMMIT"),
    )
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


@pytest.mark.asyncio
async def test_a_hung_holder_steps_down_after_the_check_timeout(
    lease_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-open connection must not wedge the holder's loop."""
    lease = AdvisoryLease(lease_name)
    try:
        assert await lease.ensure()
        holder = lease._connection  # pyright: ignore[reportPrivateUsage]
        assert holder is not None

        async def hang(*_args: Any, **_kwargs: Any) -> None:
            await asyncio.sleep(60)

        monkeypatch.setattr(scheduler_lease, "LEASE_CHECK_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(AsyncConnection, "execute", hang)

        assert not await lease.ensure()
        assert not lease.held
    finally:
        await lease.release()


@pytest.mark.asyncio
async def test_the_held_connection_sets_server_keepalives(lease_name: str) -> None:
    lease = AdvisoryLease(lease_name)
    try:
        assert await lease.ensure()
        holder = lease._connection  # pyright: ignore[reportPrivateUsage]
        assert holder is not None
        assert await holder.scalar(text("SHOW tcp_keepalives_idle")) == "30"
    finally:
        await lease.release()
