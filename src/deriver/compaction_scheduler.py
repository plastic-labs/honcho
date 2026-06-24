"""Compaction scheduler — periodically compacts the access log.

Follows the GC protocol pattern from agentc conventions:
- Proactive compaction at a good stopping point (not waiting for forced)
- Returns a gap-note style report (what was pruned, what survived, why)
- Anchors to the retention policy version for auditability
- Verifies post-compaction health

Runs as a background task in the deriver process (sibling to the
reconciler and promotion schedulers).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.crud.graph_memory import compact_access_log
from src.dependencies import tracked_db

logger = logging.getLogger(__name__)

# How often to run compaction (default: daily)
COMPACTION_INTERVAL_HOURS = 24

# How old events must be before compaction prunes them
# (5 activation half-lives = ~5 days with 24h half-life)
RETENTION_HALF_LIVES = 5
ACTIVATION_HALF_LIFE_HOURS = 24


class CompactionScheduler:
    """Background scheduler that periodically compacts the access log.
    
    Follows the GC protocol pattern:
    - Runs proactively at a fixed interval (not waiting for forced compaction)
    - Logs a gap-note style report on each run
    - Anchors to the retention policy version for auditability
    - Verifies post-compaction health
    """

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the compaction scheduler loop."""
        logger.info(
            "Starting compaction scheduler (interval: %dh, retention: %d half-lives)",
            COMPACTION_INTERVAL_HOURS, RETENTION_HALF_LIVES,
        )
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the compaction scheduler."""
        logger.info("Stopping compaction scheduler")
        self._shutdown_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main loop: compact the access log on schedule."""
        while not self._shutdown_event.is_set():
            try:
                await self._run_compaction()
            except Exception as e:
                logger.error("Compaction run failed: %s", e)
            
            # Sleep with shutdown awareness
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=COMPACTION_INTERVAL_HOURS * 3600,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Normal interval elapsed

    async def _run_compaction(self) -> None:
        """Run a single compaction cycle and log a gap-note style report."""
        logger.info("Starting compaction cycle")
        
        async with tracked_db("compaction_scheduler.run") as db:
            report = await compact_access_log(db=db)
        
        pruned = report["pruned_events"]
        if pruned > 0:
            logger.info(
                "Compaction complete: pruned %d events (%.1f%% of pre-compaction total). "
                "Retention: %d half-lives (~%d days). "
                "Post-compaction: %d events remaining. "
                "Health: %s. "
                "Note: %s",
                pruned,
                report["post_compaction"]["pruned_percentage"],
                report["retention_policy"]["half_lives"],
                report["retention_policy"]["cutoff_age_hours"] / 24,
                report["post_compaction"]["remaining_events"],
                report["health"],
                report["note"],
            )
        else:
            logger.debug("Compaction cycle: no events to prune")


# Singleton
_compaction_scheduler: CompactionScheduler | None = None


def get_compaction_scheduler() -> CompactionScheduler | None:
    return _compaction_scheduler


def set_compaction_scheduler(scheduler: CompactionScheduler) -> None:
    global _compaction_scheduler
    _compaction_scheduler = scheduler
