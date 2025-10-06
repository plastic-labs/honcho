"""Tests that QueueManager instances share the same DreamScheduler singleton."""

import asyncio
import contextlib

from src.deriver.queue_manager import QueueManager
from src.dreamer.dream_scheduler import DreamScheduler


def test_queue_manager_shares_dream_scheduler():
    """Test that multiple QueueManager instances share the same DreamScheduler."""
    # Reset singleton state
    DreamScheduler.reset_singleton()

    # Create first QueueManager
    manager1 = QueueManager()

    # Create second QueueManager
    manager2 = QueueManager()

    # Both should have the same DreamScheduler instance
    assert manager1.dream_scheduler is manager2.dream_scheduler

    # Should share the same pending_dreams dict
    assert (
        manager1.dream_scheduler.pending_dreams
        is manager2.dream_scheduler.pending_dreams
    )


async def test_queue_manager_preserves_dream_scheduler_state():
    """Test that creating a new QueueManager doesn't reset DreamScheduler state."""
    # Reset singleton state
    DreamScheduler.reset_singleton()

    # Create first QueueManager and modify scheduler state
    manager1 = QueueManager()

    # Create a dummy task to add to pending_dreams
    async def dummy_task():
        pass

    task = asyncio.create_task(dummy_task())
    manager1.dream_scheduler.pending_dreams["test_key"] = task

    # Create second QueueManager
    manager2 = QueueManager()

    # Second manager should see the first manager's state
    assert "test_key" in manager2.dream_scheduler.pending_dreams
    assert manager2.dream_scheduler.pending_dreams["test_key"] is task

    # Cleanup
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
