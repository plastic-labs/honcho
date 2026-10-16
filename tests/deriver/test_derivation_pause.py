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

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.deriver import PAUSE_EXEMPT_TASK_TYPES, not_paused_clause
from src.deriver.queue_manager import QueueManager
from src.utils.work_unit import (
    _TASK_TYPES,  # pyright: ignore[reportPrivateUsage]
)
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
    assert PAUSE_EXEMPT_TASK_TYPES <= _TASK_TYPES
    assert {
        "representation",
        "summary",
        "dream",
    } == _TASK_TYPES - PAUSE_EXEMPT_TASK_TYPES


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
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    create_queue_payload: Callable[..., Any],
    add_queue_items: Callable[
        [Sequence[tuple[dict[str, Any], int]], str, str],
        Awaitable[list[models.QueueItem]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representation batch comes back empty for a tenant paused mid-unit."""
    session, peers = sample_session_with_peers
    peer = peers[0]
    message = models.Message(
        session_name=session.name,
        workspace_name=session.workspace_name,
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
        [(payload, message.id)], session.id, session.workspace_name
    )
    tenant_id = await _register_tenant(db_session, paused=False)
    # region ai
    # Built flag-off, so the key and messages need no tenant plumbing; the claim
    # row carries the tenant, which is what the batch fetch's pause check reads.
    # endregion
    aqs = models.ActiveQueueSession(
        work_unit_key=items[0].work_unit_key, tenant_id=tenant_id
    )
    db_session.add(aqs)
    await db_session.commit()
    await db_session.refresh(aqs)
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
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
