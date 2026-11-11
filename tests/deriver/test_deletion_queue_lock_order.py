"""Session and workspace deletion take queue locks in the order a completing worker does.

A worker marking its batch done updates its queue rows, then its claim row.
Deletion used to delete the claim rows first and the queue rows second, so a
deletion that landed between the worker's two statements deadlocked with it:
each held the row the other wanted next. Deletion now takes the queue rows
first, so it waits for the worker instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.dependencies import service_db as real_service_db
from src.deriver import queue_manager
from src.deriver.queue_manager import QueueManager
from src.models import DEFAULT_TENANT_ID
from tests.deriver.conftest import independent_sessions

# Bounds on a hang, not timings: generous relative to the work involved.
LOCK_WAIT_TIMEOUT_SECONDS = 10.0
FINISH_TIMEOUT_SECONDS = 15.0


async def _seed_claimed_unit(
    db_session: AsyncSession, workspace: models.Workspace, peer: models.Peer
) -> tuple[models.Session, list[models.QueueItem], str]:
    """A session with three pending queue items in one work unit, claimed by a worker."""
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()
    work_unit_key = f"representation:{workspace.name}:{session.name}:{peer.name}"
    items: list[models.QueueItem] = []
    for seq in range(1, 4):
        message = models.Message(
            tenant_id=DEFAULT_TENANT_ID,
            session_name=session.name,
            workspace_name=workspace.name,
            peer_name=peer.name,
            content=f"message {seq}",
            token_count=1,
            seq_in_session=seq,
        )
        db_session.add(message)
        await db_session.flush()
        item = models.QueueItem(
            session_id=session.id,
            work_unit_key=work_unit_key,
            task_type="representation",
            payload={},
            processed=False,
            workspace_name=workspace.name,
            message_id=message.id,
            tenant_id=DEFAULT_TENANT_ID,
        )
        db_session.add(item)
        items.append(item)
    db_session.add(
        models.ActiveQueueSession(
            work_unit_key=work_unit_key, tenant_id=DEFAULT_TENANT_ID
        )
    )
    await db_session.commit()
    return session, items, work_unit_key


async def _wait_until_waiting_on_a_lock(db: AsyncSession, pid: int) -> None:
    """Return once backend ``pid`` is blocked on a lock."""
    deadline = asyncio.get_running_loop().time() + LOCK_WAIT_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        waiting = await db.scalar(
            text(
                "SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid = :pid"
            ),
            {"pid": pid},
        )
        await db.commit()
        if waiting:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"backend {pid} never waited on a lock")


@pytest.fixture
def pause_after_first_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the worker's completion transaction open between its two statements."""
    first_statement_done = asyncio.Event()
    resume = asyncio.Event()

    @asynccontextmanager
    async def paused_service_db(
        operation_name: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[AsyncSession]:
        async with real_service_db(operation_name, **kwargs) as db:
            if operation_name == "process_queue_item_batch":
                execute = db.execute
                statements = 0

                async def execute_then_pause(*args: Any, **kw: Any) -> Any:
                    nonlocal statements
                    result = await execute(*args, **kw)
                    statements += 1
                    if statements == 1:
                        first_statement_done.set()
                        await resume.wait()
                    return result

                db.execute = execute_then_pause
            yield db

    monkeypatch.setattr(queue_manager, "service_db", paused_service_db)
    return first_statement_done, resume


Deletion = Callable[[AsyncSession, models.Session], Coroutine[Any, Any, object]]


async def _delete_session(db: AsyncSession, session: models.Session) -> object:
    return await crud.delete_session(
        db, workspace_name=session.workspace_name, session_name=session.name
    )


async def _delete_workspace(db: AsyncSession, session: models.Session) -> object:
    return await crud.delete_workspace(db, session.workspace_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delete", [_delete_session, _delete_workspace], ids=["session", "workspace"]
)
async def test_deletion_waits_for_a_completing_worker_instead_of_deadlocking(
    delete: Deletion,
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    pause_after_first_statement: tuple[asyncio.Event, asyncio.Event],
) -> None:
    workspace, peer = sample_data
    session, items, work_unit_key = await _seed_claimed_unit(
        db_session, workspace, peer
    )
    first_statement_done, resume = pause_after_first_statement
    sessions = independent_sessions(db_engine)

    # The worker marks its batch done and stops after its queue-row update,
    # holding those row locks.
    completion = asyncio.create_task(
        QueueManager().mark_queue_items_as_processed(items, work_unit_key)
    )
    await asyncio.wait_for(first_statement_done.wait(), LOCK_WAIT_TIMEOUT_SECONDS)

    async with sessions() as deleter, sessions() as observer:
        deleter_pid = await deleter.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(deleter_pid, int)
        deletion = asyncio.create_task(delete(deleter, session))
        # The deletion blocks on the worker's queue rows. Before the fix it had
        # already deleted the worker's claim row by then.
        await _wait_until_waiting_on_a_lock(observer, deleter_pid)

        # The worker's claim-row update is next. With the old order it waited
        # on the deletion while the deletion waited on it.
        resume.set()
        await asyncio.wait_for(completion, FINISH_TIMEOUT_SECONDS)
        await asyncio.wait_for(deletion, FINISH_TIMEOUT_SECONDS)

        remaining_items = await observer.scalar(
            select(func.count())
            .select_from(models.QueueItem)
            .where(models.QueueItem.work_unit_key == work_unit_key)
        )
        remaining_claims = await observer.scalar(
            select(func.count())
            .select_from(models.ActiveQueueSession)
            .where(models.ActiveQueueSession.work_unit_key == work_unit_key)
        )
    assert remaining_items == 0
    assert remaining_claims == 0
