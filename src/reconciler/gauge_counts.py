"""Bounded counts for the reconciler's backlog gauges."""

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

# region ai
# Every replica running the reconciler scheduler refreshes these gauges, so an
# unbounded count costs the backlog size times the replica count each interval,
# and the backlog is largest exactly when something is wrong. A capped count
# reads at most GAUGE_COUNT_CAP + 1 index entries: the gauge then reads the cap
# and its *_capped companion reads 1 ("at least this many").
# endregion
GAUGE_COUNT_CAP = 100_000


async def capped_count(
    db: AsyncSession, rows: Select[Any], cap: int = GAUGE_COUNT_CAP
) -> tuple[int, bool]:
    """Count `rows` up to `cap`; returns (min(count, cap), whether the count hit the cap)."""
    counted = await db.scalar(
        select(func.count()).select_from(rows.limit(cap + 1).subquery())
    )
    count = int(counted or 0)
    return min(count, cap), count > cap
