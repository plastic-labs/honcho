"""The shared final-write step: capture, parent locks, collection check, claim check.

Each test plays one producer by hand: capture a snapshot, let a deletion land,
then open ``final_write``. Producer-specific tests live with their producers.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import models
from src.crud.lifecycle_barrier import (
    ClaimStamp,
    begin_session_deletion,
    begin_workspace_deletion,
    bump_collection_generations,
    capture_write_snapshot,
    claim_context,
    current_claim,
    ensure_collections,
    final_write,
    stamp_representations,
)
from src.exceptions import (
    WriteClaimLostError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)
from tests.deriver.conftest import independent_sessions
from tests.lifecycle_barrier_graph import (
    FINISH_TIMEOUT_SECONDS,
    MODELS,
    Graph,
    capture,
    document,
    document_count,
    seed_graph,
    tombstone,
    wait_until_waiting_on_a_lock,
    write,
)


async def test_capture_uses_stable_ids_and_generations(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    assert snapshot.tenant_id == graph.tenant_id
    assert snapshot.workspace.uid == graph.workspace.id
    assert snapshot.workspace.deletion_generation == 0
    assert snapshot.session is not None
    assert snapshot.session.uid == graph.session.id
    assert {p.uid for p in snapshot.peers} == {graph.observer.id, graph.observed.id}
    assert [c.uid for c in snapshot.collections] == [graph.collection.id]
    assert snapshot.message_ids == (graph.message_public_id,)


@pytest.mark.parametrize("missing", ["workspace", "session", "peer"])
async def test_capture_refuses_a_parent_that_is_not_there(
    db_session: AsyncSession, missing: str
) -> None:
    graph = await seed_graph(db_session)
    kwargs: dict[str, Any] = {}
    workspace_name = graph.workspace.name
    if missing == "workspace":
        workspace_name = "no-such-workspace"
    elif missing == "session":
        kwargs["session_name"] = "no-such-session"
    else:
        kwargs["peer_names"] = ["no-such-peer"]

    with pytest.raises(WriteParentGoneError) as raised:
        await capture_write_snapshot(db_session, workspace_name, **kwargs)
    assert raised.value.scope == missing


@pytest.mark.parametrize("kind", ["workspace", "session", "observer", "observed"])
async def test_capture_refuses_a_tombstoned_parent(
    db_session: AsyncSession, kind: str
) -> None:
    graph = await seed_graph(db_session)
    await tombstone(db_session, graph, kind)

    with pytest.raises(WriteParentGoneError):
        await capture(db_session, graph)


async def test_unchanged_input_passes_and_the_step_never_commits(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    async with final_write(db_session, snapshot):
        db_session.add(document(graph))
    # The step left the transaction open: another connection sees nothing yet.
    assert db_session.in_transaction()
    async with independent_sessions(db_engine)() as other:
        assert await document_count(other, graph) == 0
    await db_session.commit()
    assert await document_count(db_session, graph) == 1


@pytest.mark.parametrize("kind", ["workspace", "session"])
async def test_a_raised_generation_is_stale(
    db_session: AsyncSession, kind: str
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    model = MODELS[kind]
    await db_session.execute(
        update(model)
        .where(model.id == getattr(graph, kind).id)
        .values(deletion_generation=model.deletion_generation + 1)
    )
    await db_session.commit()

    with pytest.raises(WriteGenerationStaleError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == kind
    assert await document_count(db_session, graph) == 0


@pytest.mark.parametrize("kind", ["workspace", "session", "observer", "observed"])
async def test_a_tombstoned_parent_is_gone(db_session: AsyncSession, kind: str) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    await tombstone(db_session, graph, kind)

    with pytest.raises(WriteParentGoneError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == ("peer" if kind.startswith("obs") else kind)
    assert await document_count(db_session, graph) == 0


async def test_a_session_recreated_under_the_same_name_does_not_pass(
    db_session: AsyncSession,
) -> None:
    """The snapshot holds the session id, so a new row with the old name fails."""
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph, message_ids=[])

    await db_session.execute(
        delete(models.Message).where(models.Message.id == graph.message_id)
    )
    await db_session.execute(
        delete(models.Session).where(models.Session.id == graph.session.id)
    )
    db_session.add(
        models.Session(name=graph.session.name, workspace_name=graph.workspace.name)
    )
    await db_session.commit()

    with pytest.raises(WriteParentGoneError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "session"
    assert await document_count(db_session, graph) == 0


async def test_a_deleted_source_message_is_stale(db_session: AsyncSession) -> None:
    """A context-only source has no queue row, so only the barrier can see it go."""
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    await db_session.execute(
        update(models.Message)
        .where(models.Message.id == graph.message_id)
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.commit()

    with pytest.raises(WriteGenerationStaleError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "message"
    assert await document_count(db_session, graph) == 0


async def test_a_collection_deletion_is_caught_after_the_document_writes(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    await bump_collection_generations(
        db_session, graph.workspace.name, [(graph.observer.name, graph.observed.name)]
    )
    await db_session.commit()

    wrote = False
    with pytest.raises(WriteGenerationStaleError) as raised:
        async with final_write(db_session, snapshot):
            db_session.add(document(graph))
            # The parent check passed, so the producer's writes ran.
            wrote = True
    await db_session.rollback()
    assert wrote
    assert raised.value.scope == "collection"
    assert await document_count(db_session, graph) == 0


async def test_a_collection_created_after_capture_passes_only_at_generation_zero(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session, with_collection=False)
    snapshot = await capture(db_session, graph)
    assert [c.uid for c in snapshot.collections] == [None]
    pair = (graph.observer.name, graph.observed.name)

    async with final_write(db_session, snapshot):
        await ensure_collections(db_session, snapshot, [pair])
        db_session.add(document(graph))
    await db_session.commit()
    assert await document_count(db_session, graph) == 1

    # A deletion in the new collection, then a second write from the old snapshot.
    await bump_collection_generations(db_session, graph.workspace.name, [pair])
    await db_session.commit()
    with pytest.raises(WriteGenerationStaleError):
        await write(db_session, graph, snapshot)
    assert await document_count(db_session, graph) == 1


async def test_stamp_representations_adds_peers_and_collections(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture_write_snapshot(
        db_session, graph.workspace.name, session_name=graph.session.name
    )
    assert snapshot.peers == ()

    pair = (graph.observer.name, graph.observed.name)
    stamped = await stamp_representations(db_session, snapshot, [pair])
    assert {p.name for p in stamped.peers} == set(pair)
    assert [c.uid for c in stamped.collections] == [graph.collection.id]

    narrowed = stamped.for_representation(graph.observed.name, graph.observed.name)
    assert [p.name for p in narrowed.peers] == [graph.observed.name]
    assert narrowed.collections == ()


async def _seed_claim(
    db: AsyncSession, graph: Graph, tenant_id: str | None
) -> ClaimStamp:
    work_unit_key = (
        f"representation:{graph.workspace.name}:{graph.session.name}:observed"
    )
    item = models.QueueItem(
        session_id=graph.session.id,
        work_unit_key=work_unit_key,
        task_type="representation",
        payload={},
        workspace_name=graph.workspace.name,
        message_id=graph.message_id,
        tenant_id=tenant_id,
    )
    claim = models.ActiveQueueSession(work_unit_key=work_unit_key, tenant_id=tenant_id)
    db.add_all([item, claim])
    await db.commit()
    return ClaimStamp(work_unit_key, claim.id, tenant_id, (item.id,))


@pytest.mark.parametrize("tenant_id", [None, models.DEFAULT_TENANT_ID])
async def test_a_held_claim_passes_with_and_without_a_tenant(
    db_session: AsyncSession, tenant_id: str | None
) -> None:
    """Flag-off queue and claim rows carry a NULL tenant; they still match."""
    graph = await seed_graph(db_session)
    claim = await _seed_claim(db_session, graph, tenant_id)
    snapshot = await capture(db_session, graph, claim=claim)

    await write(db_session, graph, snapshot)
    assert await document_count(db_session, graph) == 1


@pytest.mark.parametrize("removed", ["claim", "queue"])
async def test_a_revoked_claim_or_removed_work_is_refused(
    db_session: AsyncSession, removed: str
) -> None:
    graph = await seed_graph(db_session)
    claim = await _seed_claim(db_session, graph, None)
    snapshot = await capture(db_session, graph, claim=claim)

    table = models.ActiveQueueSession if removed == "claim" else models.QueueItem
    await db_session.execute(
        delete(table).where(table.work_unit_key == claim.work_unit_key)
    )
    await db_session.commit()

    with pytest.raises(WriteClaimLostError):
        await write(db_session, graph, snapshot)
    assert await document_count(db_session, graph) == 0


@pytest.mark.parametrize(
    ("row_tenant", "stamp_tenant"),
    [("tenant-b", None), (None, "tenant-a"), ("tenant-b", "tenant-a")],
)
async def test_another_tenants_claim_on_the_same_key_does_not_pass(
    db_session: AsyncSession, row_tenant: str | None, stamp_tenant: str | None
) -> None:
    """A key that reads the same for two tenants must not stand in for ours."""
    graph = await seed_graph(db_session)
    theirs = await _seed_claim(db_session, graph, row_tenant)
    ours = replace(theirs, tenant_id=stamp_tenant)
    snapshot = await capture(db_session, graph, claim=ours)

    with pytest.raises(WriteClaimLostError):
        await write(db_session, graph, snapshot)
    assert await document_count(db_session, graph) == 0
    # The check only reads: their claim and their queued work are untouched.
    assert await db_session.scalar(
        select(func.count()).select_from(models.ActiveQueueSession)
    )
    assert await db_session.scalar(select(func.count()).select_from(models.QueueItem))


async def test_claim_context_binds_the_work_units_claim() -> None:
    assert current_claim() is None
    claim = ClaimStamp("representation:w:s:p", "claim-id", None)
    async with claim_context(claim):
        narrowed = current_claim([3, 1, 3])
        assert narrowed is not None
        assert narrowed.claim_id == "claim-id"
        assert narrowed.queue_item_ids == (1, 3)
    assert current_claim() is None


async def _begin_deletion(db: AsyncSession, graph: Graph, kind: str) -> None:
    if kind == "workspace":
        await begin_workspace_deletion(db, graph.workspace.name)
    else:
        await begin_session_deletion(db, graph.workspace.name, graph.session.name)


@pytest.mark.parametrize("kind", ["workspace", "session"])
async def test_a_deletion_waits_for_an_open_final_write(
    db_session: AsyncSession, db_engine: AsyncEngine, kind: str
) -> None:
    """The write got its locks first, so it commits and the deletion follows it."""
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)
    sessions = independent_sessions(db_engine)

    async with sessions() as producer, sessions() as deleter, sessions() as watch:
        # No commit after this: the deletion must run on the same connection.
        deleter_pid = await deleter.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(deleter_pid, int)
        deletion: asyncio.Task[None] | None = None
        try:
            async with final_write(producer, snapshot):
                producer.add(document(graph))
                deletion = asyncio.create_task(_begin_deletion(deleter, graph, kind))
                await wait_until_waiting_on_a_lock(watch, deleter_pid)
            await producer.commit()

            await asyncio.wait_for(deletion, FINISH_TIMEOUT_SECONDS)
            await deleter.commit()
            # The deletion ran after the write, so the write is its to remove.
            assert await document_count(watch, graph) == 1
        finally:
            # On a failure, free the locks so the blocked task can end.
            await producer.rollback()
            if deletion is not None:
                await asyncio.wait({deletion}, timeout=FINISH_TIMEOUT_SECONDS)


@pytest.mark.parametrize("kind", ["workspace", "session"])
async def test_a_final_write_waits_for_an_open_deletion_and_is_then_refused(
    db_session: AsyncSession, db_engine: AsyncEngine, kind: str
) -> None:
    """The deletion got its lock first, so the write waits, then sees it."""
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)
    sessions = independent_sessions(db_engine)

    async with sessions() as producer, sessions() as deleter, sessions() as watch:
        # No commit after this: the write must run on the same connection.
        producer_pid = await producer.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(producer_pid, int)
        pending: asyncio.Task[None] | None = None
        try:
            await _begin_deletion(deleter, graph, kind)
            pending = asyncio.create_task(write(producer, graph, snapshot))
            await wait_until_waiting_on_a_lock(watch, producer_pid)
            await deleter.commit()

            with pytest.raises(WriteGenerationStaleError) as raised:
                await asyncio.wait_for(pending, FINISH_TIMEOUT_SECONDS)
            assert raised.value.scope == kind
            assert await document_count(watch, graph) == 0
        finally:
            await deleter.rollback()
            if pending is not None:
                await asyncio.wait({pending}, timeout=FINISH_TIMEOUT_SECONDS)


async def test_a_write_under_several_sessions_is_refused_when_any_was_deleted(
    db_session: AsyncSession,
) -> None:
    """A public write of several conclusions can name several sessions."""
    graph = await seed_graph(db_session)
    other = models.Session(name="other", workspace_name=graph.workspace.name)
    db_session.add(other)
    await db_session.commit()
    other_id = other.id

    snapshot = await capture(db_session, graph, session_names=["other"], message_ids=[])
    assert {s.uid for s in snapshot.sessions} == {graph.session.id, other_id}
    with pytest.raises(ValueError, match="more than one session"):
        _ = snapshot.session

    await write(db_session, graph, snapshot)
    assert await document_count(db_session, graph) == 1

    await begin_session_deletion(db_session, graph.workspace.name, "other")
    await db_session.commit()
    with pytest.raises(WriteGenerationStaleError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "session"
    assert await document_count(db_session, graph) == 1
