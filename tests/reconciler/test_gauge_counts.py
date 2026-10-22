"""The reconciler's backlog gauges count up to a cap and say when they hit it."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.reconciler.gauge_counts import capped_count


@pytest.mark.asyncio
async def test_capped_count_stops_at_the_cap(db_session: AsyncSession) -> None:
    prefix = f"cap-{generate_nanoid(size=8)}"
    db_session.add_all(models.Workspace(name=f"{prefix}-{i}") for i in range(5))
    await db_session.commit()
    rows = select(models.Workspace.id).where(models.Workspace.name.like(f"{prefix}%"))

    assert await capped_count(db_session, rows, cap=3) == (3, True)
    assert await capped_count(db_session, rows, cap=5) == (5, False)
    assert await capped_count(db_session, rows, cap=10) == (5, False)
