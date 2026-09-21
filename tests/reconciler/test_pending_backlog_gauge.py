"""The pending-embeddings backlog gauge must be refreshed per-replica.

These tests pin both halves: the scheduler loop drives the refresh, and the
queue-driven reconciliation cycle does not.
"""
# region ai
# ``message_embeddings_pending`` reports a DB-global count, so it is the one gauge
# here whose value is service-wide rather than per-process. It is also zero-
# initialized at startup, which makes a missing refresh actively harmful: a replica
# that never measured the backlog would export a confident, permanently-healthy 0.
# So the count is driven from ``ReconcilerScheduler._scheduler_loop`` (runs on every
# replica, every interval), NOT from ``run_vector_reconciliation_cycle`` (runs off
# the queue behind work-unit dedup, so exactly one replica per cycle executes it).
# endregion

import asyncio

import pytest

from src.reconciler import scheduler as scheduler_module
from src.reconciler import sync_vectors
from src.reconciler.scheduler import ReconcilerScheduler


@pytest.fixture(autouse=True)
def _reset_scheduler_singleton():  # pyright: ignore[reportUnusedFunction]
    ReconcilerScheduler.reset_singleton()
    yield
    ReconcilerScheduler.reset_singleton()


async def test_scheduler_loop_refreshes_backlog_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every scheduler iteration refreshes the gauge, on every replica.

    Patched at the scheduler's own reference so this asserts the call site, not
    just that the function exists.
    """
    calls = 0
    refreshed = asyncio.Event()

    async def _fake_refresh() -> None:
        nonlocal calls
        calls += 1
        refreshed.set()

    # region ai
    # Patched onto the class, so it is invoked as a bound method — it needs the
    # ``self`` parameter or the call raises TypeError, which ``_scheduler_loop`` would
    # then swallow, leaving this guard silently inert.
    # endregion
    async def _never_enqueue(_self: object, _task: object) -> bool:
        return False

    monkeypatch.setattr(
        scheduler_module, "record_pending_embeddings_backlog", _fake_refresh
    )
    monkeypatch.setattr(
        ReconcilerScheduler, "_try_enqueue_task", _never_enqueue, raising=True
    )

    scheduler = ReconcilerScheduler()
    await scheduler.start()
    try:
        await asyncio.wait_for(refreshed.wait(), timeout=5.0)
    finally:
        await scheduler.shutdown()

    assert calls >= 1, "scheduler loop never refreshed the backlog gauge"


def test_reconciliation_cycle_does_not_drive_the_gauge() -> None:
    """The queue-driven cycle must not be the thing that sets the gauge."""
    # region ai
    # If the refresh moves back into ``run_vector_reconciliation_cycle``, only the
    # replica that wins the ``sync_vectors`` work unit would ever measure the backlog,
    # and the zero-init would go back to lying on all the others.
    #
    # Structural guard: the cycle is a long DB-driven coroutine, so this inspects the
    # global names it references rather than executing it.
    # endregion
    assert hasattr(sync_vectors, "record_pending_embeddings_backlog")

    referenced = sync_vectors.run_vector_reconciliation_cycle.__code__.co_names
    assert "record_pending_embeddings_backlog" not in referenced
