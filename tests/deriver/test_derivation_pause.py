"""The derivation pause: which work it stops, and that it stops a unit already held.

The pause is ``tenants.derivation_paused``, read in SQL by
``crud.deriver.not_paused_clause`` wherever the deriver takes work. It stops
billable work only (task types outside ``PAUSE_EXEMPT_TASK_TYPES``), and it is
checked again on every in-unit fetch, so a unit claimed before the pause is
released at its next fetch rather than drained. The claim-side behaviour (skipped
not drained, gauges agree, resume on the next claim) is pinned in
``test_fair_scheduling.TestThePauseSeam``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.deriver import PAUSE_EXEMPT_TASK_TYPES, not_paused_clause
from src.db import tenant_context
from src.deriver.queue_manager import QueueManager, WorkerOwnership
from src.utils.work_unit import TASK_TYPES
from tests.deriver.conftest import SeedWorkUnit


async def _register_tenant(db: AsyncSession, *, paused: bool) -> str:
    """A tenants row with an id unique to this test (the rows outlive the table clear)."""
    tenant_id = f"t-pause-{generate_nanoid(size=8)}"
    db.add(models.Tenant(tenant_id=tenant_id, tier="shared", derivation_paused=paused))
    await db.commit()
    return tenant_id


async def _set_paused(db: AsyncSession, tenant_id: str, *, paused: bool) -> None:
    tenant = await db.get(models.Tenant, tenant_id)
    assert tenant is not None
    tenant.derivation_paused = paused
    await db.commit()


# ---------------------------------------------------------------------------
# Which work the pause stops
# ---------------------------------------------------------------------------


def test_every_task_type_is_classified() -> None:
    """Adding a task type forces a decision: exempt it here, or accept it pauses.

    The exemption list defaults a new type to paused, which is the safe direction
    for billing. This pins the current split so the default is never silent.
    """
    assert PAUSE_EXEMPT_TASK_TYPES <= TASK_TYPES
    assert {
        "representation",
        "summary",
        "dream",
    } == TASK_TYPES - PAUSE_EXEMPT_TASK_TYPES


def test_the_clause_is_never_built_flag_off() -> None:
    assert (
        not_paused_clause(
            models.QueueItemBatch.tenant_id, models.QueueItemBatch.task_type
        )
        is None
    )
    assert not_paused_clause(models.QueueItem.tenant_id, "summary") is None


def test_a_known_exempt_task_type_needs_no_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert not_paused_clause(models.QueueItem.tenant_id, "deletion") is None
    assert not_paused_clause(models.QueueItem.tenant_id, "summary") is not None


@pytest.mark.asyncio
async def test_a_paused_tenants_exempt_work_still_claims(
    db_session: AsyncSession,
    seed_work_unit: SeedWorkUnit,
    batch_gate_settings: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion and webhooks run for a paused tenant; its billable work does not."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    batch_gate_settings(workers=10)
    tenant_id = await _register_tenant(db_session, paused=True)
    deletion_key = f"{tenant_id}:deletion:workspace:session:resource"
    webhook_key = f"{tenant_id}:webhook:workspace"
    summary_key = f"{tenant_id}:summary:workspace:session:peer:peer"
    await seed_work_unit(
        deletion_key, task_type="deletion", tenant_id=tenant_id, with_messages=False
    )
    await seed_work_unit(
        webhook_key, task_type="webhook", tenant_id=tenant_id, with_messages=False
    )
    await seed_work_unit(
        summary_key, task_type="summary", tenant_id=tenant_id, with_messages=False
    )

    claimed = await QueueManager().get_and_claim_work_units()

    assert set(claimed) == {deletion_key, webhook_key}


# ---------------------------------------------------------------------------
# A pause that lands while a unit is held
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_next_item_fetch_stops_once_the_tenant_is_paused(
    db_session: AsyncSession,
    seed_work_unit: SeedWorkUnit,
    batch_gate_settings: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker's loop ends on an empty fetch, which releases the unit."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    batch_gate_settings(workers=10)
    tenant_id = await _register_tenant(db_session, paused=False)
    summary_key = f"{tenant_id}:summary:workspace:session:peer:peer"
    deletion_key = f"{tenant_id}:deletion:workspace:session:resource"
    await seed_work_unit(
        summary_key,
        task_type="summary",
        tenant_id=tenant_id,
        token_counts=(1, 1),
        with_messages=False,
    )
    await seed_work_unit(
        deletion_key, task_type="deletion", tenant_id=tenant_id, with_messages=False
    )
    qm = QueueManager()
    claimed = await qm.get_and_claim_work_units()
    assert set(claimed) == {summary_key, deletion_key}

    assert await qm.get_next_queue_item("summary", summary_key, claimed[summary_key])

    await _set_paused(db_session, tenant_id, paused=True)
    assert (
        await qm.get_next_queue_item("summary", summary_key, claimed[summary_key])
        is None
    )
    # Exempt work keeps flowing for the same tenant.
    assert await qm.get_next_queue_item("deletion", deletion_key, claimed[deletion_key])

    await _set_paused(db_session, tenant_id, paused=False)
    assert await qm.get_next_queue_item("summary", summary_key, claimed[summary_key])


@pytest.mark.asyncio
async def test_the_next_batch_fetch_stops_once_the_tenant_is_paused(
    db_session: AsyncSession,
    create_queue_payload: Callable[..., Any],
    add_queue_items: Callable[
        [Sequence[tuple[dict[str, Any], int]], str, str],
        Awaitable[list[models.QueueItem]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representation batch comes back empty for a tenant paused mid-unit."""
    tenant_id = await _register_tenant(db_session, paused=False)
    # region ai
    # Built flag-on under the tenant, as a real writer would: the rows are
    # stamped with it and the key carries its prefix. The batch fetch refuses
    # an un-prefixed key flag-on, so a flag-off key can't stand in here.
    # endregion
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(tenant_id)
    try:
        workspace = models.Workspace(name=str(generate_nanoid()))
        db_session.add(workspace)
        await db_session.flush()
        peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        db_session.add_all([peer, session])
        await db_session.flush()
        message = models.Message(
            session_name=session.name,
            workspace_name=workspace.name,
            peer_name=peer.name,
            content="hello",
            token_count=10,
            seq_in_session=1,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        payload = create_queue_payload(
            message=message,
            task_type="representation",
            observed=peer.name,
            observer=peer.name,
        )
        items = await add_queue_items(
            [(payload, message.id)], session.id, workspace.name
        )
    finally:
        tenant_context.reset(token)
    assert items[0].work_unit_key.startswith(f"{tenant_id}:")
    assert items[0].tenant_id == tenant_id
    aqs = models.ActiveQueueSession(
        work_unit_key=items[0].work_unit_key, tenant_id=tenant_id
    )
    db_session.add(aqs)
    await db_session.commit()
    await db_session.refresh(aqs)
    qm = QueueManager()

    batch = await qm.get_queue_item_batch(
        task_type="representation",
        work_unit_key=items[0].work_unit_key,
        aqs_id=aqs.id,
    )
    assert [item.id for item in batch.items_to_process] == [items[0].id]

    await _set_paused(db_session, tenant_id, paused=True)
    paused_batch = await qm.get_queue_item_batch(
        task_type="representation",
        work_unit_key=items[0].work_unit_key,
        aqs_id=aqs.id,
    )
    assert paused_batch.items_to_process == []
    assert paused_batch.messages_context == []


@pytest.mark.asyncio
async def test_a_pause_mid_unit_is_not_reported_as_a_drain(
    db_session: AsyncSession,
    seed_work_unit: SeedWorkUnit,
    batch_gate_settings: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unit is released, but queue.empty is not sent while items are suspended."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    batch_gate_settings(workers=10)
    tenant_id = await _register_tenant(db_session, paused=False)
    summary_key = f"{tenant_id}:summary:workspace:session:peer:peer"
    await seed_work_unit(
        summary_key,
        task_type="summary",
        tenant_id=tenant_id,
        token_counts=(1, 1),
        with_messages=False,
    )
    qm = QueueManager()
    claimed = await qm.get_and_claim_work_units()
    qm.worker_ownership["w"] = WorkerOwnership(
        work_unit_key=summary_key, aqs_id=claimed[summary_key]
    )

    async def pause_after_first_item(_item: Any) -> None:
        await _set_paused(db_session, tenant_id, paused=True)

    publish = AsyncMock()
    with (
        patch(
            "src.deriver.queue_manager.process_item",
            side_effect=pause_after_first_item,
        ),
        patch("src.deriver.queue_manager.publish_webhook_event", publish),
    ):
        await qm.process_work_unit(summary_key, "w")

    publish.assert_not_awaited()
    db_session.expire_all()
    pending = await db_session.scalar(
        select(func.count())
        .select_from(models.QueueItem)
        .where(models.QueueItem.work_unit_key == summary_key)
        .where(~models.QueueItem.processed)
    )
    assert pending == 1
    held = await db_session.scalar(
        select(func.count())
        .select_from(models.ActiveQueueSession)
        .where(models.ActiveQueueSession.work_unit_key == summary_key)
    )
    assert held == 0


@pytest.mark.asyncio
async def test_a_drain_flag_on_still_reports_queue_empty(
    db_session: AsyncSession,
    seed_work_unit: SeedWorkUnit,
    batch_gate_settings: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    batch_gate_settings(workers=10)
    tenant_id = await _register_tenant(db_session, paused=False)
    summary_key = f"{tenant_id}:summary:workspace:session:peer:peer"
    await seed_work_unit(
        summary_key,
        task_type="summary",
        tenant_id=tenant_id,
        token_counts=(1, 1),
        with_messages=False,
    )
    qm = QueueManager()
    claimed = await qm.get_and_claim_work_units()
    qm.worker_ownership["w"] = WorkerOwnership(
        work_unit_key=summary_key, aqs_id=claimed[summary_key]
    )

    publish = AsyncMock()
    with (
        patch("src.deriver.queue_manager.process_item", AsyncMock()),
        patch("src.deriver.queue_manager.publish_webhook_event", publish),
    ):
        await qm.process_work_unit(summary_key, "w")

    publish.assert_awaited_once()
