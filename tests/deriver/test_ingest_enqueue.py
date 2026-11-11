"""The ingest enqueuer: pending records become the message route's queue work."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import models, schemas
from src.backlog import outstanding_work_seconds
from src.crud import ingest as ingest_crud
from src.crud.deriver import get_deriver_metrics
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.deriver import ingest_enqueue
from src.deriver.ingest_enqueue import enqueue_session, sweep_pending_ingest


def _workspace(client: TestClient) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name})
    assert response.status_code in (200, 201), response.text
    return name


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": "discord",
        "source_instance": "guild1",
        "source_id": f"msg-{generate_nanoid(size=8)}",
        "created_at": "2026-11-04T20:47:00Z",
        "container": {"id": "chan-1", "name": "dev"},
        "author": {"id": "user-1", "name": "Jessica"},
        "content": "Northstar needs SSO before rollout.",
    }
    record.update(overrides)
    return record


def _ingest(
    client: TestClient, workspace: str, *records: dict[str, Any], **body: Any
) -> list[dict[str, Any]]:
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest", json={"records": list(records), **body}
    )
    assert response.status_code == 200, response.text
    return response.json()["records"]


async def _queue_rows(
    db: AsyncSession, workspace: str, session: str
) -> list[tuple[str, list[str], str | None]]:
    """(task_type, observers, observed) for a session's queue rows, sorted."""
    session_id = await db.scalar(
        select(models.Session.id).where(
            models.Session.workspace_name == workspace,
            models.Session.name == session,
        )
    )
    rows = (
        await db.execute(
            select(models.QueueItem.task_type, models.QueueItem.payload).where(
                models.QueueItem.workspace_name == workspace,
                models.QueueItem.session_id == session_id,
            )
        )
    ).all()
    return sorted(
        (
            task_type,
            sorted(payload.get("observers") or []),
            payload.get("observed"),
        )
        for task_type, payload in rows
    )


async def _derivations(db: AsyncSession, workspace: str) -> list[str]:
    rows = await db.execute(
        select(models.IngestRecord.derivation)
        .where(models.IngestRecord.workspace_name == workspace)
        .order_by(models.IngestRecord.derivation)
    )
    return list(rows.scalars())


@pytest.mark.asyncio
async def test_ingested_messages_get_the_message_routes_queue_work(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Same peers, same messages: ingest queues what the message route queues."""
    workspace = _workspace(client)
    for session in ("via-messages", "via-ingest"):
        response = client.post(
            f"/v3/workspaces/{workspace}/sessions",
            json={
                "id": session,
                "peers": {"alice": {}, "bob": {"observe_others": True}},
            },
        )
        assert response.status_code in (200, 201), response.text
    contents = ["first", "second", "third"]
    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/via-messages/messages",
        json={"messages": [{"peer_id": "alice", "content": c} for c in contents]},
    )
    assert response.status_code in (200, 201), response.text

    results = _ingest(
        client,
        workspace,
        *(
            _record(
                content=c,
                created_at=f"2026-11-04T20:4{i}:00Z",
                session_id="via-ingest",
                peer_id="alice",
            )
            for i, c in enumerate(contents)
        ),
    )
    assert {r["derivation"] for r in results} == {"queued"}

    via_messages = await _queue_rows(db_session, workspace, "via-messages")
    via_ingest = await _queue_rows(db_session, workspace, "via-ingest")
    assert via_messages  # the comparison is not vacuous
    assert via_ingest == via_messages
    assert await _derivations(db_session, workspace) == ["queued"] * 3


@pytest.mark.asyncio
async def test_derive_later_queues_nothing(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    [result] = _ingest(client, workspace, _record(), derive="later")
    assert result["derivation"] == "deferred"
    queue = await db_session.scalar(
        select(func.count())
        .select_from(models.QueueItem)
        .where(models.QueueItem.workspace_name == workspace)
    )
    assert queue == 0


@pytest.mark.asyncio
async def test_records_with_reasoning_off_get_no_representation_work(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    [result] = _ingest(
        client,
        workspace,
        _record(configuration={"reasoning": {"enabled": False}}),
    )
    assert result["derivation"] == "queued"
    rows = await _queue_rows(db_session, workspace, result["session_id"])
    assert all(task_type != "representation" for task_type, _, _ in rows)


@pytest.mark.asyncio
async def test_a_fenced_session_is_left_for_the_sweep(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held fence skips the session; the sweep queues it once the fence is free."""
    workspace = sample_data[0].name
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate({"records": [_record()]})
        [created] = (await ingest_crud.ingest_batch(db, workspace, batch)).results
    assert created.session_id is not None

    async with factory() as holder, factory() as db:
        await take_workspace_fence(holder, workspace, mode="shared")
        await take_session_fences(holder, workspace, [created.session_id])
        outcome = await enqueue_session(db, workspace, created.session_id)
        assert (outcome.skipped, outcome.queued) == (True, 0)
        assert await _derivations(db, workspace) == ["pending"]
        await holder.rollback()

    monkeypatch.setattr(
        ingest_enqueue, "SWEEP_MIN_AGE", ingest_enqueue.SWEEP_MIN_AGE * 0
    )
    swept = await sweep_pending_ingest()
    assert swept.queued >= 1
    async with factory() as db:
        assert await _derivations(db, workspace) == ["queued"]
        assert await _queue_rows(db, workspace, created.session_id)


@pytest.mark.asyncio
async def test_a_record_whose_message_is_gone_is_dropped(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace = sample_data[0].name
    batch = schemas.IngestBatch.model_validate({"records": [_record()]})
    [created] = (await ingest_crud.ingest_batch(db_session, workspace, batch)).results
    assert created.session_id is not None
    await db_session.execute(
        delete(models.Message).where(models.Message.public_id == created.message_id)
    )
    await db_session.commit()

    outcome = await enqueue_session(db_session, workspace, created.session_id)
    assert (outcome.queued, outcome.orphaned) == (0, 1)
    assert await _derivations(db_session, workspace) == []


@pytest.mark.asyncio
async def test_the_sweep_has_nothing_to_do_without_old_pending_records(
    db_session: AsyncSession,
) -> None:
    assert not await ingest_enqueue.has_pending_ingest(db_session)


def test_a_repeat_in_the_same_batch_is_reported_queued_too(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    record = _record()
    first, repeat = _ingest(client, workspace, record, record)
    assert (first["outcome"], repeat["outcome"]) == ("created", "duplicate")
    assert first["derivation"] == repeat["derivation"] == "queued"


@pytest.mark.asyncio
async def test_the_sweep_gets_past_a_session_it_cannot_queue(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one session per run, a fenced session doesn't take every run."""
    workspace = sample_data[0].name
    monkeypatch.setattr(
        ingest_enqueue, "SWEEP_MIN_AGE", ingest_enqueue.SWEEP_MIN_AGE * 0
    )
    monkeypatch.setattr(ingest_enqueue, "_SWEEP_SESSIONS", 1)
    monkeypatch.setattr(ingest_enqueue, "_sweep_cursor", None)
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate(
            {
                "records": [
                    _record(source_id="a", session_id="a-session"),
                    _record(source_id="b", session_id="b-session"),
                ]
            }
        )
        await ingest_crud.ingest_batch(db, workspace, batch)

    derivation: str | None = None
    async with factory() as holder:
        await take_workspace_fence(holder, workspace, mode="shared")
        await take_session_fences(holder, workspace, ["a-session"])
        for _ in range(20):
            await sweep_pending_ingest()
            async with factory() as db:
                derivation = await db.scalar(
                    select(models.IngestRecord.derivation).where(
                        models.IngestRecord.workspace_name == workspace,
                        models.IngestRecord.session_name == "b-session",
                    )
                )
            if derivation == "queued":
                break
        await holder.rollback()
    assert derivation == "queued"


@pytest.mark.asyncio
async def test_pending_records_count_as_deriver_work(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stranded records keep a deriver running, since its sweep queues them."""
    workspace = sample_data[0].name
    batch = schemas.IngestBatch.model_validate({"records": [_record()]})
    await ingest_crud.ingest_batch(db_session, workspace, batch)

    stats = await get_deriver_metrics(db_session)
    assert stats.ingest_pending >= 1
    monkeypatch.setattr(
        ingest_enqueue, "SWEEP_MIN_AGE", ingest_enqueue.SWEEP_MIN_AGE * 0
    )
    stats = await get_deriver_metrics(db_session)
    assert stats.ingest_pending_due >= 1
    assert await ingest_enqueue.has_pending_ingest(db_session)
    # A record pending past the wake window still counts, but no longer as work.
    monkeypatch.setattr(
        ingest_enqueue, "SWEEP_WAKE_WINDOW", ingest_enqueue.SWEEP_WAKE_WINDOW * 0
    )
    stats = await get_deriver_metrics(db_session)
    assert stats.ingest_pending >= 1
    assert stats.ingest_pending_due == 0
    # Nor does it schedule sweeps (each a queued unit of deriver work) on its own.
    assert not await ingest_enqueue.has_pending_ingest(db_session)
    idle = schemas.DeriverMetrics()
    assert outstanding_work_seconds(idle, dreams_due=0) == 0
    assert (
        outstanding_work_seconds(
            idle.model_copy(update={"ingest_pending_due": 1}), dreams_due=0
        )
        > 0
    )


def test_the_scheduler_gates_the_sweep_on_the_wake_window() -> None:
    """The reconciler only schedules sweeps through `has_pending_ingest`."""
    from src.reconciler.scheduler import RECONCILER_TASKS

    task = RECONCILER_TASKS["enqueue_ingest"]
    assert task.has_pending is ingest_enqueue.has_pending_ingest
    assert task.payload == {"reconciler_type": "enqueue_ingest"}
