"""Derive later: POST /ingest/derive releases deferred records for derivation."""

import asyncio
import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import models, schemas
from src.config import settings
from src.crud import ingest as ingest_crud
from src.crud.deriver import get_deriver_metrics
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.deriver import ingest_enqueue
from src.deriver.ingest_enqueue import EnqueueOutcome, sweep_pending_ingest
from src.routers import ingest as ingest_router
from src.security import JWTParams, create_jwt


def _workspace(client: TestClient) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name, "auto": True})
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


def _derive(client: TestClient, workspace: str) -> dict[str, Any]:
    response = client.post(f"/v3/workspaces/{workspace}/ingest/derive")
    assert response.status_code == 200, response.text
    return response.json()


async def _derivations(db: AsyncSession, workspace: str) -> list[str]:
    rows = await db.execute(
        select(models.IngestRecord.derivation)
        .where(models.IngestRecord.workspace_name == workspace)
        .order_by(models.IngestRecord.derivation)
    )
    return list(rows.scalars())


async def _queue_shape(db: AsyncSession, workspace: str) -> list[tuple[str, int]]:
    """(task_type, rows) for the workspace's queue."""
    rows = await db.execute(
        select(models.QueueItem.task_type, func.count())
        .where(models.QueueItem.workspace_name == workspace)
        .group_by(models.QueueItem.task_type)
        .order_by(models.QueueItem.task_type)
    )
    return [(task_type, int(count)) for task_type, count in rows.all()]


# ---------------------------------------------------------------------------
# POST /ingest/derive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_derive_queues_deferred_records_like_ingesting_them_now(
    client: TestClient, db_session: AsyncSession
) -> None:
    """Released records get exactly the queue work a `derive: "now"` batch gets."""
    records = [
        _record(source_id="a"),
        _record(source_id="b", author={"id": "user-2", "name": "Aakash"}),
        _record(source_id="c", container={"id": "chan-2", "name": "ops"}),
    ]
    later = _workspace(client)
    now = _workspace(client)
    deferred = _ingest(client, later, *records, derive="later")
    assert [r["derivation"] for r in deferred] == ["deferred"] * 3
    assert await _queue_shape(db_session, later) == []
    _ingest(client, now, *records)

    assert _derive(client, later) == {"pending": 3, "remaining": 0}
    assert await _derivations(db_session, later) == ["queued"] * 3
    assert await _queue_shape(db_session, later) == await _queue_shape(db_session, now)
    # Nothing left: calling again releases nothing and queues nothing more.
    assert _derive(client, later) == {"pending": 0, "remaining": 0}
    assert await _queue_shape(db_session, later) == await _queue_shape(db_session, now)


@pytest.mark.asyncio
async def test_derive_releases_in_batches_until_none_remain(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_crud, "DERIVE_BATCH", 2)
    workspace = _workspace(client)
    _ingest(
        client,
        workspace,
        *[
            _record(source_id=f"m{i}", container={"id": f"chan-{i % 2}"})
            for i in range(5)
        ],
        derive="later",
    )

    calls = [_derive(client, workspace) for _ in range(3)]
    assert calls == [
        {"pending": 2, "remaining": 3},
        {"pending": 2, "remaining": 1},
        {"pending": 1, "remaining": 0},
    ]
    assert await _derivations(db_session, workspace) == ["queued"] * 5


@pytest.mark.asyncio
async def test_a_session_is_released_oldest_message_first(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every call leaves the released part of a session a prefix of its messages.

    Summaries cover sequence windows, and the summarizer skips a message once a
    later summary exists, so a message released after a later one would never
    be summarized.
    """
    monkeypatch.setattr(ingest_crud, "DERIVE_BATCH", 3)
    workspace = _workspace(client)
    _ingest(
        client,
        workspace,
        *[
            _record(source_id=f"m{i:02}", created_at=f"2026-11-04T20:{i:02}:00Z")
            for i in range(12)
        ],
        derive="later",
    )

    async def queued_seqs() -> list[int]:
        rows = await db_session.execute(
            select(models.Message.seq_in_session)
            .join(
                models.IngestRecord,
                models.IngestRecord.message_public_id == models.Message.public_id,
            )
            .where(
                models.IngestRecord.workspace_name == workspace,
                models.IngestRecord.derivation == "queued",
            )
            .order_by(models.Message.seq_in_session)
        )
        return list(rows.scalars())

    for call in range(1, 5):
        assert _derive(client, workspace)["pending"] == 3
        assert await queued_seqs() == list(range(1, 3 * call + 1))


@pytest.mark.asyncio
async def test_derive_leaves_records_ingested_now_alone(
    client: TestClient, db_session: AsyncSession
) -> None:
    workspace = _workspace(client)
    _ingest(client, workspace, _record(source_id="now"))
    queued_before = await _queue_shape(db_session, workspace)
    _ingest(
        client,
        workspace,
        _record(source_id="later", container={"id": "chan-2"}),
        derive="later",
    )

    assert _derive(client, workspace) == {"pending": 1, "remaining": 0}
    assert await _derivations(db_session, workspace) == ["queued", "queued"]
    # The new session's work, and nothing again for the first record.
    after = dict(await _queue_shape(db_session, workspace))
    assert all(after[task] == 2 * count for task, count in queued_before)


def test_derive_on_a_missing_workspace_is_404(client: TestClient) -> None:
    response = client.post("/v3/workspaces/no-such-workspace/ingest/derive")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_records_released_long_after_ingest_still_wake_the_sweep(
    client: TestClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How long a record has been pending runs from its release, not its ingest.

    A record ingested two days ago and released now, whose enqueue found its
    session busy, must still count as deriver work and schedule the sweep.
    """
    workspace = _workspace(client)
    _ingest(client, workspace, _record(), derive="later")
    two_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=2)
    await db_session.execute(
        update(models.IngestRecord)
        .where(models.IngestRecord.workspace_name == workspace)
        .values(created_at=two_days_ago, derivation_changed_at=two_days_ago)
    )
    await db_session.commit()

    async def busy(*_args: Any, **_kwargs: Any) -> EnqueueOutcome:
        return EnqueueOutcome(skipped=True)

    monkeypatch.setattr(ingest_router, "enqueue_session", busy)
    assert _derive(client, workspace) == {"pending": 1, "remaining": 0}
    assert await _derivations(db_session, workspace) == ["pending"]
    # Just released: the sweep leaves it to the request that released it.
    monkeypatch.setattr(ingest_enqueue, "_sweep_cursor", None)
    assert (await sweep_pending_ingest()).sessions == 0
    assert await _derivations(db_session, workspace) == ["pending"]

    monkeypatch.setattr(
        ingest_enqueue, "SWEEP_MIN_AGE", ingest_enqueue.SWEEP_MIN_AGE * 0
    )
    due = await db_session.scalar(
        select(func.count()).where(
            models.IngestRecord.workspace_name == workspace,
            ingest_enqueue.sweep_due_clause(),
        )
    )
    assert due == 1
    assert await ingest_enqueue.has_pending_ingest(db_session)
    assert (await get_deriver_metrics(db_session)).ingest_pending_due >= 1

    monkeypatch.setattr(ingest_enqueue, "_sweep_cursor", None)
    await sweep_pending_ingest()
    assert await _derivations(db_session, workspace) == ["queued"]


@pytest.mark.asyncio
async def test_derive_waits_for_a_session_fence(
    db_engine: AsyncEngine, auto_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A batch or deletion holding the session's fence holds off the release."""
    workspace = auto_data[0].name
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate(
            {"derive": "later", "records": [_record()]}
        )
        [created] = (await ingest_crud.ingest_batch(db, workspace, batch)).results
    assert created.session_id is not None

    async with factory() as holder, factory() as db:
        await take_workspace_fence(holder, workspace, mode="shared")
        await take_session_fences(holder, workspace, [created.session_id])
        release = asyncio.create_task(ingest_crud.release_deferred(db, workspace))
        await asyncio.sleep(0.5)
        assert not release.done()
        fence_freed_at = await holder.scalar(select(func.clock_timestamp()))
        await holder.rollback()
        released = await release
    assert (released.pending, released.sessions) == (1, [created.session_id])
    # Pending since the release, not since the derive call began waiting.
    async with factory() as db:
        changed_at = await db.scalar(
            select(models.IngestRecord.derivation_changed_at).where(
                models.IngestRecord.workspace_name == workspace
            )
        )
    assert fence_freed_at is not None and changed_at is not None
    assert changed_at >= fence_freed_at


@pytest.mark.asyncio
async def test_concurrent_derive_calls_release_each_record_once(
    db_engine: AsyncEngine, auto_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace = auto_data[0].name
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate(
            {
                "derive": "later",
                "records": [
                    _record(source_id=f"m{i}", container={"id": f"chan-{i % 3}"})
                    for i in range(9)
                ],
            }
        )
        await ingest_crud.ingest_batch(db, workspace, batch)

    async def release() -> int:
        async with factory() as db:
            return (await ingest_crud.release_deferred(db, workspace)).pending

    first, second = await asyncio.gather(release(), release())
    assert first + second == 9
    async with factory() as db:
        assert await _derivations(db, workspace) == ["pending"] * 9
        assert await ingest_crud.count_deferred(db, workspace) == 0


# ---------------------------------------------------------------------------
# Fence order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolved", "raced"), [("aaa-early", True), ("zzz-late", False)]
)
async def test_a_session_bound_meanwhile_is_fenced_only_in_sorted_order(
    db_session: AsyncSession,
    auto_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
    resolved: str,
    raced: bool,
) -> None:
    """A batch never takes a fence that sorts before one it holds.

    Here a concurrent batch has bound the record's container, after this batch
    read its bindings, to a session this batch hasn't fenced. Every other fence
    holder (a derive call, a deletion, another batch) takes fences in sorted
    order, so taking one that sorts before a held fence could close a cycle:
    the batch starts over instead, and its retry reads the new binding before
    taking any fence. A name that sorts after every held fence is taken in order.
    """
    workspace = auto_data[0].name

    async def bound_meanwhile(*_args: Any, **_kwargs: Any) -> tuple[str, bool]:
        return resolved, False

    monkeypatch.setattr(ingest_crud, "_session_name", bound_meanwhile)
    batch = schemas.IngestBatch.model_validate({"records": [_record()]})
    if raced:
        with pytest.raises(ingest_crud.IngestRaced):
            await ingest_crud.ingest_batch(db_session, workspace, batch)
        await db_session.rollback()
    else:
        [created] = (
            await ingest_crud.ingest_batch(db_session, workspace, batch)
        ).results
        assert (created.outcome, created.session_id) == ("created", resolved)


# ---------------------------------------------------------------------------
# Auth: workspace and admin keys only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("ingest", {"records": [_record()]}),
        ("ingest/derive", None),
    ],
)
def test_only_workspace_keys_reach_the_ingest_routes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: dict[str, Any] | None,
) -> None:
    workspace = _workspace(client)
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    url = f"/v3/workspaces/{workspace}/{path}"

    for refused in (
        JWTParams(w=workspace, p="someone"),
        JWTParams(w=workspace, s="some-session"),
        JWTParams(w="another-workspace"),
    ):
        client.headers["Authorization"] = f"Bearer {create_jwt(refused)}"
        assert client.post(url, json=body).status_code == 401, refused

    client.headers["Authorization"] = f"Bearer {create_jwt(JWTParams(w=workspace))}"
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
