"""Shutdown must bound the drain and give claims back before the process dies.

A claim still held at exit strands its work until the stale sweep reclaims it.
"""

import asyncio
import signal
from unittest.mock import AsyncMock, patch

import pytest

from src.deriver.queue_manager import QueueManager


@pytest.mark.asyncio
async def test_drain_is_bounded_and_claims_are_released() -> None:
    """A task outliving the budget is abandoned; cleanup still runs."""
    qm = QueueManager()

    started = asyncio.Event()

    async def never_finishes() -> None:
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(never_finishes())
    qm.add_task(task)
    await started.wait()

    with (
        patch.object(qm, "cleanup", new=AsyncMock()) as cleanup,
        patch.object(qm.dream_scheduler, "shutdown", new=AsyncMock()),
        patch.object(qm.reconciler_scheduler, "shutdown", new=AsyncMock()),
        patch("src.deriver.queue_manager.settings") as settings,
    ):
        settings.DERIVER.SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 0.05

        loop = asyncio.get_running_loop()
        began = loop.time()
        await qm.shutdown(signal.SIGTERM)
        elapsed = loop.time() - began

    assert elapsed < 2, f"shutdown blocked on the stuck task for {elapsed:.1f}s"
    assert qm.shutdown_event.is_set()
    cleanup.assert_awaited_once()  # claims released before exit

    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_fast_work_still_completes_before_claims_are_released() -> None:
    """Bounding the drain must not cut short work that fits."""
    qm = QueueManager()
    finished = False

    async def quick() -> None:
        nonlocal finished
        await asyncio.sleep(0.01)
        finished = True

    qm.add_task(asyncio.create_task(quick()))

    with (
        patch.object(qm, "cleanup", new=AsyncMock()) as cleanup,
        patch.object(qm.dream_scheduler, "shutdown", new=AsyncMock()),
        patch.object(qm.reconciler_scheduler, "shutdown", new=AsyncMock()),
        patch("src.deriver.queue_manager.settings") as settings,
    ):
        settings.DERIVER.SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 5.0
        await qm.shutdown(signal.SIGTERM)

    assert finished
    cleanup.assert_awaited_once()
