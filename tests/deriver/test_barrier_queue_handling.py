"""How the queue worker treats a write the lifecycle barrier refused.

A refusal is not a failure of the work. The worker drops the item when its
parent is gone, releases the unit when it lost its claim, and retries the item
(which recomputes it from live input) when only a generation moved. It also
binds its claim around every item, so producers can prove at their final write
that the worker still holds the unit.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.lifecycle_barrier import ClaimStamp, current_claim
from src.deriver import queue_manager
from src.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS, QueueManager
from src.exceptions import (
    LifecycleBarrierError,
    WriteClaimLostError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)
from src.utils.queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue_manager, "RETRY_BACKOFF_SECONDS", 0.0)


async def _seed_dream_item(
    db: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> tuple[str, int]:
    workspace, peer = sample_data
    key = f"dream:omni:{workspace.name}:{peer.name}:{peer.name}"
    item = models.QueueItem(
        work_unit_key=key,
        task_type="dream",
        payload={"dream_type": "omni", "observer": peer.name, "observed": peer.name},
        workspace_name=workspace.name,
    )
    db.add(item)
    await db.commit()
    return key, item.id


async def _process(db: AsyncSession, key: str, outcome: Any) -> list[ClaimStamp | None]:
    """Claim the unit and run it once; ``process_item`` raises or returns ``outcome``."""
    claims: list[ClaimStamp | None] = []

    async def process_item(_item: models.QueueItem) -> None:
        claims.append(current_claim())
        if isinstance(outcome, BaseException):
            raise outcome

    manager = QueueManager()
    claimed = await manager.claim_work_units(db, [key])
    await db.commit()
    manager.track_worker_work_unit("worker", key, claimed[key])
    with patch.object(queue_manager, "process_item", process_item):
        await manager.process_work_unit(key, "worker")
    for claim in claims:
        assert claim is not None
        assert claim.claim_id == claimed[key]
    return claims


async def _item(db: AsyncSession, item_id: int) -> tuple[bool, str | None, int]:
    row = (
        await db.execute(
            select(
                models.QueueItem.processed,
                models.QueueItem.error,
                models.QueueItem.payload,
            ).where(models.QueueItem.id == item_id)
        )
    ).one()
    return row.processed, row.error, row.payload.get(RETRY_ATTEMPTS_PAYLOAD_KEY, 0)


async def test_the_worker_binds_its_claim_around_each_item(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    key, item_id = await _seed_dream_item(db_session, sample_data)

    [claim] = await _process(db_session, key, None)

    assert claim is not None
    assert claim.work_unit_key == key
    assert claim.queue_item_ids == (item_id,)
    # Flag off, queue and claim rows carry no tenant; the stamp says the same.
    assert claim.tenant_id is None
    assert current_claim() is None
    assert await _item(db_session, item_id) == (True, None, 0)


async def test_a_deleted_parent_drops_the_item_without_an_error(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    key, item_id = await _seed_dream_item(db_session, sample_data)

    await _process(
        db_session, key, WriteParentGoneError("peer", "the observed peer was deleted")
    )

    assert await _item(db_session, item_id) == (True, None, 0)


async def test_a_lost_claim_leaves_the_item_for_the_worker_that_holds_it(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    key, item_id = await _seed_dream_item(db_session, sample_data)

    claims = await _process(
        db_session, key, WriteClaimLostError("claim", "the claim was revoked")
    )

    # Tried once, then released: not retried in place, not counted, not failed.
    assert len(claims) == 1
    assert await _item(db_session, item_id) == (False, None, 0)


async def test_a_stale_write_is_retried_and_then_fails_like_a_transient_error(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    key, item_id = await _seed_dream_item(db_session, sample_data)
    stale: LifecycleBarrierError = WriteGenerationStaleError(
        "collection", "a deletion landed"
    )

    for attempt in range(1, MAX_RETRYABLE_ATTEMPTS):
        await _process(db_session, key, stale)
        assert await _item(db_session, item_id) == (False, None, attempt)

    # A unit that is stale every time does not loop forever.
    await _process(db_session, key, stale)
    processed, error, _ = await _item(db_session, item_id)
    assert processed
    assert error is not None
    assert "WriteGenerationStaleError" in error


async def test_a_recomputed_item_completes(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    key, item_id = await _seed_dream_item(db_session, sample_data)
    await _process(
        db_session, key, WriteGenerationStaleError("session", "a deletion landed")
    )
    assert await _item(db_session, item_id) == (False, None, 1)

    await _process(db_session, key, None)
    processed, error, _ = await _item(db_session, item_id)
    assert (processed, error) == (True, None)


async def test_a_unit_whose_session_is_gone_is_dropped_at_the_batch_fetch(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """Queued representation work for a session that no longer exists."""
    workspace, peer = sample_data
    key = f"representation:{workspace.name}:{generate_nanoid()}:{peer.name}"
    item = models.QueueItem(
        work_unit_key=key,
        task_type="representation",
        payload={"observers": [peer.name], "observed": peer.name},
        workspace_name=workspace.name,
    )
    db_session.add(item)
    await db_session.commit()

    manager = QueueManager()
    claimed = await manager.claim_work_units(db_session, [key])
    await db_session.commit()
    manager.track_worker_work_unit("worker", key, claimed[key])
    with patch.object(queue_manager, "process_representation_batch") as deriver:
        await manager.process_work_unit(key, "worker")

    deriver.assert_not_called()
    assert await _item(db_session, item.id) == (True, None, 0)
