from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from src import models
from src.reconciler import queue_cleanup


@pytest.mark.asyncio
async def test_cleanup_queue_items_executes_delete_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = AsyncMock()
    called_operation_names: list[str | None] = []

    @asynccontextmanager
    async def fake_tracked_db(operation_name: str | None = None):
        called_operation_names.append(operation_name)
        yield fake_db

    monkeypatch.setattr(queue_cleanup, "tracked_db", fake_tracked_db)

    await queue_cleanup.cleanup_queue_items()

    assert called_operation_names == ["cleanup_queue_items"]
    fake_db.execute.assert_awaited_once()
    fake_db.commit.assert_awaited_once()

    delete_stmt = fake_db.execute.await_args.args[0]
    assert delete_stmt.table.name == models.QueueItem.__tablename__
    assert delete_stmt.whereclause is not None
