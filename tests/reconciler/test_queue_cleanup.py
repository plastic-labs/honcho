from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.reconciler import queue_cleanup


def _queue_item(
    key: str,
    *,
    processed: bool,
    error: str | None = None,
    created_at: datetime | None = None,
) -> models.QueueItem:
    item = models.QueueItem(
        work_unit_key=key,
        payload={},
        session_id=None,
        task_type="reconciler",
        workspace_name=None,
        message_id=None,
        processed=processed,
        error=error,
    )
    if created_at is not None:
        item.created_at = created_at
    return item


@pytest.mark.asyncio
async def test_cleanup_queue_items_deletes_in_batches(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    called_operation_names: list[str | None] = []

    @asynccontextmanager
    async def _db(operation_name: str | None = None) -> AsyncGenerator[AsyncSession]:
        called_operation_names.append(operation_name)
        yield db_session

    monkeypatch.setattr(queue_cleanup, "service_db", _db)
    monkeypatch.setattr(queue_cleanup, "QUEUE_CLEANUP_BATCH_SIZE", 2)

    executes = 0
    real_execute = db_session.execute

    async def counting_execute(*args: Any, **kwargs: Any) -> Any:
        nonlocal executes
        executes += 1
        return await real_execute(*args, **kwargs)

    monkeypatch.setattr(db_session, "execute", counting_execute)

    old = datetime.now(UTC) - timedelta(
        seconds=settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS + 60
    )
    db_session.add_all(
        [
            *(_queue_item(f"reconciler:done-{i}", processed=True) for i in range(4)),
            _queue_item(
                "reconciler:old-error", processed=True, error="x", created_at=old
            ),
            _queue_item("reconciler:pending", processed=False),
            _queue_item("reconciler:new-error", processed=True, error="x"),
        ]
    )
    await db_session.commit()

    deleted = await queue_cleanup.cleanup_queue_items()

    assert deleted == 5
    assert executes == 3
    assert called_operation_names == ["cleanup_queue_items"]
    remaining = (
        (await real_execute(select(models.QueueItem.work_unit_key))).scalars().all()
    )
    assert sorted(remaining) == ["reconciler:new-error", "reconciler:pending"]
