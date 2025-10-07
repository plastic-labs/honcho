"""Tests for DreamScheduler singleton pattern."""

import asyncio
import contextlib

from src.dreamer.dream_scheduler import DreamScheduler


def test_dream_scheduler_singleton():
    """Test that DreamScheduler implements proper singleton pattern."""
    # Reset singleton state
    DreamScheduler.reset_singleton()

    # Create first instance
    scheduler1 = DreamScheduler()

    # Create second instance
    scheduler2 = DreamScheduler()

    # Both should be the same instance
    assert scheduler1 is scheduler2

    # Should share the same pending_dreams dict
    assert scheduler1.pending_dreams is scheduler2.pending_dreams


async def test_dream_scheduler_initialized_once():
    """Test that DreamScheduler is only initialized once."""
    # Reset singleton state
    DreamScheduler.reset_singleton()

    # Create first instance
    scheduler1 = DreamScheduler()

    # Create a dummy task to add to pending_dreams
    async def dummy_task():
        pass

    task = asyncio.create_task(dummy_task())
    scheduler1.pending_dreams["test_key"] = task

    # Create second instance
    scheduler2 = DreamScheduler()

    # Second instance should have the same data as first
    assert "test_key" in scheduler2.pending_dreams
    assert scheduler2.pending_dreams["test_key"] is task

    # Cleanup
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_dream_scheduler_multiple_instances():
    """Test that creating multiple instances doesn't reset state."""
    # Reset singleton state
    DreamScheduler.reset_singleton()

    instances = [DreamScheduler() for _ in range(5)]

    # All instances should be the same
    for instance in instances[1:]:
        assert instance is instances[0]
