"""get_queue_item_batch fails closed on an un-prefixed key flag-on."""

from typing import Any

import pytest

from src.config import settings
from src.deriver import queue_manager as qm_module


def _no_db(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("get_queue_item_batch opened a DB session")


@pytest.mark.asyncio
async def test_unprefixed_key_raises_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(qm_module, "service_db", _no_db)
    qm = qm_module.QueueManager()
    with pytest.raises(ValueError, match="no tenant prefix"):
        await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key="representation:default:s1:alice",
            aqs_id="aqs",
        )
