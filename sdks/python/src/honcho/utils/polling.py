"""Queue polling utilities for the Honcho Python SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..api_types import QueueStatusResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=QueueStatusResponse)


def poll_until_complete(
    get_status: Callable[[], T],
    timeout: float = 300.0,
) -> T:
    """
    Poll queue status until all work units are complete.

    This utility function handles the polling logic for waiting until
    pending_work_units and in_progress_work_units are both 0.

    Args:
        get_status: A callable that returns the current QueueStatusResponse
        timeout: Maximum time to poll in seconds (default: 300 seconds / 5 minutes)

    Returns:
        QueueStatusResponse when all work units are complete

    Raises:
        TimeoutError: If timeout is exceeded before work units complete
    """
    start_time = time.time()

    while True:
        try:
            status = get_status()
        except Exception as e:
            logger.warning("Failed to get queue status: %s", e)
            time.sleep(1)

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Error during status check: {e}"
                ) from e
            continue

        if status.pending_work_units == 0 and status.in_progress_work_units == 0:
            return status

        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            raise TimeoutError(
                f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress work units."
            )

        # Sleep for the expected time to complete all current work units
        # Assuming each pending and in-progress work unit takes 1 second
        total_work_units = status.pending_work_units + status.in_progress_work_units
        sleep_time = max(1, total_work_units)

        # Don't sleep past the timeout
        remaining_time = timeout - elapsed_time
        sleep_time = min(sleep_time, remaining_time)
        if sleep_time <= 0:
            raise TimeoutError(
                f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress work units."
            )

        time.sleep(sleep_time)


async def poll_until_complete_async(
    get_status: Callable[[], Awaitable[T]],
    timeout: float = 300.0,
) -> T:
    """
    Poll queue status until all work units are complete (async version).

    This utility function handles the polling logic for waiting until
    pending_work_units and in_progress_work_units are both 0.

    Args:
        get_status: An async callable that returns the current QueueStatusResponse
        timeout: Maximum time to poll in seconds (default: 300 seconds / 5 minutes)

    Returns:
        QueueStatusResponse when all work units are complete

    Raises:
        TimeoutError: If timeout is exceeded before work units complete
    """
    start_time = time.time()

    while True:
        try:
            status = await get_status()
        except Exception as e:
            logger.warning("Failed to get queue status: %s", e)
            await asyncio.sleep(1)

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Error during status check: {e}"
                ) from e
            continue

        if status.pending_work_units == 0 and status.in_progress_work_units == 0:
            return status

        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            raise TimeoutError(
                f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress work units."
            )

        total_work_units = status.pending_work_units + status.in_progress_work_units
        sleep_time = max(1, total_work_units)
        remaining_time = timeout - elapsed_time
        sleep_time = min(sleep_time, remaining_time)

        if sleep_time <= 0:
            raise TimeoutError(
                f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress work units."
            )

        await asyncio.sleep(sleep_time)
