"""The message routes commit each message together with the queue rows that derive it.

A message is never saved without its work: a failed queue insert fails the
write, and a deadlock reruns the whole transaction. Faults are injected at
``stage_enqueue``, the step that inserts the queue rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.deriver.enqueue import stage_enqueue
from src.routers.messages import (
    _create_and_enqueue,  # pyright: ignore[reportPrivateUsage]
)

MESSAGES = 3


class _PgError(Exception):
    sqlstate: str

    def __init__(self, sqlstate: str) -> None:
        super().__init__(sqlstate)
        self.sqlstate = sqlstate


def _db_error(sqlstate: str) -> DBAPIError:
    return DBAPIError("INSERT INTO queue", {}, _PgError(sqlstate))


def _failing_first(
    sqlstate: str, *, times: int
) -> tuple[Callable[..., Awaitable[None]], list[int]]:
    """``stage_enqueue`` that raises ``sqlstate`` on its first ``times`` calls."""
    calls: list[int] = []

    async def stage(
        db: AsyncSession, payload: list[dict[str, Any]], **kwargs: Any
    ) -> None:
        calls.append(len(payload))
        if len(calls) <= times:
            raise _db_error(sqlstate)
        await stage_enqueue(db, payload, **kwargs)

    return stage, calls


def _post(client: TestClient, workspace: str, session: str, peer: str) -> Any:
    return client.post(
        f"/v3/workspaces/{workspace}/sessions/{session}/messages",
        json={
            "messages": [
                {"peer_id": peer, "content": f"message {i}"} for i in range(MESSAGES)
            ]
        },
    )


async def _saved(
    db: AsyncSession, workspace: str, session: str
) -> tuple[list[int], int]:
    """Sequence numbers of the session's messages, and its representation rows."""
    await db.rollback()
    seqs = (
        await db.scalars(
            select(models.Message.seq_in_session)
            .where(
                models.Message.workspace_name == workspace,
                models.Message.session_name == session,
            )
            .order_by(models.Message.seq_in_session)
        )
    ).all()
    queued = await db.scalar(
        select(func.count())
        .select_from(models.QueueItem)
        .join(models.Message, models.Message.id == models.QueueItem.message_id)
        .where(
            models.Message.workspace_name == workspace,
            models.Message.session_name == session,
            models.QueueItem.task_type == "representation",
        )
    )
    return list(seqs), queued or 0


@pytest.mark.asyncio
async def test_a_deadlock_reruns_the_whole_write(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """The rerun inserts the messages again; the lost attempt leaves nothing."""
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session = str(generate_nanoid())
    stage, calls = _failing_first("40P01", times=1)

    with patch("src.routers.messages.stage_enqueue", stage):
        response = _post(client, workspace_name, session, peer_name)

    assert response.status_code == 201, response.text
    assert calls == [MESSAGES, MESSAGES]
    seqs, queued = await _saved(db_session, workspace_name, session)
    assert seqs == list(range(1, MESSAGES + 1))
    assert queued == MESSAGES


@pytest.mark.asyncio
@pytest.mark.parametrize("upload", [False, True])
async def test_a_deadlock_after_the_queue_insert_reruns_into_a_new_session(
    client: TestClient,
    db_session: AsyncSession,
    upload: bool,
) -> None:
    """The first attempt creates the workspace, session and members, then loses
    the race after its queue insert. Nothing it read may outlive the rollback:
    the rerun creates them again and commits."""
    workspace_name, session = str(generate_nanoid()), str(generate_nanoid())
    calls: list[int] = []

    async def stage(
        db: AsyncSession, payload: list[dict[str, Any]], **kwargs: Any
    ) -> None:
        calls.append(len(payload))
        await stage_enqueue(db, payload, **kwargs)
        if len(calls) == 1:
            raise _db_error("40P01")

    url = f"/v3/workspaces/{workspace_name}/sessions/{session}/messages"
    with patch("src.routers.messages.stage_enqueue", stage):
        if upload:
            response = client.post(
                f"{url}/upload",
                files={"file": ("note.txt", b"hello", "text/plain")},
                data={"peer_id": "alice"},
            )
        else:
            response = client.post(
                url, json={"messages": [{"peer_id": "alice", "content": "hello"}]}
            )

    assert response.status_code == 201, response.text
    assert len(calls) == 2
    assert await _saved(db_session, workspace_name, session) == ([1], 1)
    again = client.get(f"/v3/workspaces/{workspace_name}/sessions/{session}/peers")
    assert [p["id"] for p in again.json()["items"]] == ["alice"]


@pytest.mark.asyncio
async def test_a_lock_timeout_fails_the_write_without_a_retry(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """A timeout means the session is already busy; rerunning would multiply the wait."""
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session = str(generate_nanoid())
    stage, calls = _failing_first("55P03", times=3)

    with (
        patch("src.routers.messages.stage_enqueue", stage),
        pytest.raises(DBAPIError),
    ):
        _post(client, workspace_name, session, peer_name)

    assert calls == [MESSAGES]
    assert await _saved(db_session, workspace_name, session) == ([], 0)


async def _wait_until_blocked_by(
    engine: AsyncEngine, waiter_pid: int, holder_pid: int, timeout: float = 10
) -> None:
    """Return once backend ``waiter_pid`` waits on a lock ``holder_pid`` holds."""
    async with engine.connect() as conn:
        for _ in range(int(timeout / 0.05)):
            blocked = await conn.scalar(
                text("SELECT :holder = ANY(pg_blocking_pids(:waiter))"),
                {"holder": holder_pid, "waiter": waiter_pid},
            )
            await conn.rollback()
            if blocked:
                return
            await asyncio.sleep(0.05)
    pytest.fail("the deletion never waited on the message write")


async def _messages_without_one_representation_row(
    db: AsyncSession, workspace: str, session: str
) -> int:
    """Saved messages of the session that don't have exactly one representation row."""
    per_message = (
        select(func.count())
        .select_from(models.QueueItem)
        .where(
            models.QueueItem.message_id == models.Message.id,
            models.QueueItem.task_type == "representation",
        )
        .scalar_subquery()
    )
    return (
        await db.scalar(
            select(func.count())
            .select_from(models.Message)
            .where(
                models.Message.workspace_name == workspace,
                models.Message.session_name == session,
                per_message != 1,
            )
        )
        or 0
    )


@pytest.mark.asyncio
async def test_a_message_write_racing_a_session_deletion_never_loses_its_work(
    client: TestClient,
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """Deletion takes queue locks first and deletes the session row last, so it
    can deadlock with a message write that holds the session row's key-share lock
    and then inserts queue rows. Postgres aborts one side. Either way, no message
    is left without its queue rows."""
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    session = str(generate_nanoid())
    # Pending queue rows for the session, so deletion has queue locks to take.
    assert _post(client, workspace_name, session, peer_name).status_code == 201

    sessions = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    inserted, release, deleting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pids: dict[str, int] = {}

    async def backend_pid(db: AsyncSession) -> int:
        pid = await db.scalar(select(func.pg_backend_pid()))
        assert pid is not None
        return pid

    async def paused_stage(
        db: AsyncSession, payload: list[dict[str, Any]], **kwargs: Any
    ) -> None:
        if not inserted.is_set():
            pids["writer"] = await backend_pid(db)
            inserted.set()
            await release.wait()
        await stage_enqueue(db, payload, **kwargs)

    async def write() -> None:
        async with sessions() as db:
            await _create_and_enqueue(
                db,
                BackgroundTasks(),
                workspace_name=workspace_name,
                session_name=session,
                messages=[schemas.MessageCreate(content="late", peer_id=peer_name)],
                configurations=[None],
            )

    async def delete() -> None:
        async with sessions() as db:
            pids["deleter"] = await backend_pid(db)
            deleting.set()
            await crud.delete_session(db, workspace_name, session)

    with patch("src.routers.messages.stage_enqueue", paused_stage):
        writer = asyncio.create_task(write())
        await asyncio.wait_for(inserted.wait(), timeout=10)
        deleter = asyncio.create_task(delete())
        await asyncio.wait_for(deleting.wait(), timeout=10)
        # The deletion holds the session's queue locks and waits to delete the
        # session row the paused write references.
        await _wait_until_blocked_by(db_engine, pids["deleter"], pids["writer"])
        release.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(writer, deleter, return_exceptions=True), timeout=30
        )

    # The write reruns if it loses the deadlock, so it ends committed either
    # way. A deletion that loses fails with 40P01, and the deriver retries it.
    write_outcome, delete_outcome = outcomes
    assert write_outcome is None, write_outcome
    if delete_outcome is not None:
        assert isinstance(delete_outcome, DBAPIError), delete_outcome
        assert getattr(delete_outcome.orig, "sqlstate", None) == "40P01"

    assert (
        await _messages_without_one_representation_row(
            db_session, workspace_name, session
        )
        == 0
    ), "a message was saved without exactly one representation row"
    orphans = await db_session.scalar(
        select(func.count())
        .select_from(models.QueueItem)
        .where(
            models.QueueItem.workspace_name == workspace_name,
            models.QueueItem.message_id.is_not(None),
            ~select(models.Message.id)
            .where(models.Message.id == models.QueueItem.message_id)
            .exists(),
        )
    )
    assert orphans == 0, "a queue row points at a message that was never saved"
