"""Which task types count toward the processed-items metric."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.deriver import queue_manager
from src.deriver.queue_manager import QueueManager
from src.telemetry import prometheus_metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("work_unit_key", "counted_as"),
    [
        ("representation:ws:session:observer:observed", "representation"),
        ("summary:ws:session:observer:observed", "summary"),
        ("dream:omni:ws:observer:observed", "dream"),
        ("deletion:ws:session:name", "deletion"),
        ("tenant:dream:omni:ws:observer:observed", "dream"),
        ("webhook:ws", None),
        ("reconciler:sync_vectors", None),
    ],
)
async def test_processed_items_metric_by_task_type(
    work_unit_key: str, counted_as: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def fake_service_db(_name: str) -> AsyncGenerator[AsyncMock]:
        yield AsyncMock()

    record = MagicMock()
    monkeypatch.setattr(queue_manager, "service_db", fake_service_db)
    monkeypatch.setattr(settings.METRICS, "ENABLED", True)
    monkeypatch.setattr(prometheus_metrics, "record_deriver_queue_item", record)

    items: Any = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    await QueueManager().mark_queue_items_as_processed(items, work_unit_key)

    if counted_as is None:
        record.assert_not_called()
    else:
        record.assert_called_once_with(
            count=2, workspace_name="ws", task_type=counted_as
        )
