"""A session clone that races a deletion of its source.

A clone reads the source session and writes a new one in one transaction. A
clone that committed stands on its own: deleting the source later leaves it
alone. A clone still running must not copy from a session being deleted.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.crud.lifecycle_barrier import begin_session_deletion
from src.exceptions import ResourceNotFoundException
from tests.deriver.conftest import independent_sessions
from tests.lifecycle_barrier_graph import (
    FINISH_TIMEOUT_SECONDS,
    Graph,
    seed_graph,
    wait_until_waiting_on_a_lock,
)

MESSAGES = 3


async def _seed_source(db: AsyncSession) -> Graph:
    """A source session with three messages and one member."""
    graph = await seed_graph(db)
    for seq in range(2, MESSAGES + 1):
        db.add(
            models.Message(
                workspace_name=graph.workspace.name,
                session_name=graph.session.name,
                peer_name=graph.observed.name,
                content=f"message {seq}",
                seq_in_session=seq,
            )
        )
    db.add(
        models.SessionPeer(
            workspace_name=graph.workspace.name,
            session_name=graph.session.name,
            peer_name=graph.observed.name,
        )
    )
    await db.commit()
    return graph


async def _sessions(db: AsyncSession, graph: Graph) -> dict[str, int]:
    """Each session in the workspace -> how many messages it holds."""
    rows = await db.execute(
        select(models.Session.name, func.count(models.Message.id))
        .outerjoin(
            models.Message,
            (models.Message.tenant_id == models.Session.tenant_id)
            & (models.Message.workspace_name == models.Session.workspace_name)
            & (models.Message.session_name == models.Session.name),
        )
        .where(models.Session.workspace_name == graph.workspace.name)
        .group_by(models.Session.name)
    )
    return {row[0]: row[1] for row in rows}


async def test_a_completed_clone_outlives_the_deletion_of_its_source(
    db_session: AsyncSession,
) -> None:
    graph = await _seed_source(db_session)
    clone = await crud.clone_session(
        db_session, graph.workspace.name, graph.session.name
    )
    clone_name = clone.name

    await crud.delete_session(db_session, graph.workspace.name, graph.session.name)

    assert await _sessions(db_session, graph) == {clone_name: MESSAGES}


async def test_a_deletion_waits_for_a_running_clone(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The clone locked the source first, so it copies everything and then stands alone."""
    graph = await _seed_source(db_session)
    sessions = independent_sessions(db_engine)

    async with sessions() as cloner, sessions() as deleter, sessions() as watch:
        deleter_pid = await deleter.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(deleter_pid, int)
        deletions: list[asyncio.Task[Any]] = []

        # Pause the clone after it created the new session, before it copies.
        flush = cloner.flush

        async def flush_then_let_the_deletion_start(*args: Any, **kwargs: Any) -> None:
            await flush(*args, **kwargs)
            if not deletions:
                deletions.append(
                    asyncio.create_task(
                        crud.delete_session(
                            deleter, graph.workspace.name, graph.session.name
                        )
                    )
                )
                await wait_until_waiting_on_a_lock(watch, deleter_pid)

        cloner.flush = flush_then_let_the_deletion_start  # type: ignore[method-assign]
        try:
            clone = await crud.clone_session(
                cloner, graph.workspace.name, graph.session.name
            )
            [deletion] = deletions
            await asyncio.wait_for(deletion, FINISH_TIMEOUT_SECONDS)
            assert await _sessions(watch, graph) == {clone.name: MESSAGES}
        finally:
            await cloner.rollback()
            if deletions:
                await asyncio.wait(deletions, timeout=FINISH_TIMEOUT_SECONDS)


async def test_a_clone_waits_for_a_running_deletion_and_copies_nothing(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The deletion locked the source first. The clone must not copy what is going."""
    graph = await _seed_source(db_session)
    sessions = independent_sessions(db_engine)

    async with sessions() as cloner, sessions() as deleter, sessions() as watch:
        cloner_pid = await cloner.scalar(text("SELECT pg_backend_pid()"))
        assert isinstance(cloner_pid, int)
        clone: asyncio.Task[models.Session] | None = None
        try:
            # The deletion's first statement, held open.
            await begin_session_deletion(
                deleter, graph.workspace.name, graph.session.name
            )
            clone = asyncio.create_task(
                crud.clone_session(cloner, graph.workspace.name, graph.session.name)
            )
            await wait_until_waiting_on_a_lock(watch, cloner_pid)
            # The rest of the same deletion, in the same transaction.
            await crud.delete_session(deleter, graph.workspace.name, graph.session.name)

            with pytest.raises(ResourceNotFoundException):
                await asyncio.wait_for(clone, FINISH_TIMEOUT_SECONDS)
            assert await _sessions(watch, graph) == {}
        finally:
            await deleter.rollback()
            if clone is not None:
                await asyncio.wait({clone}, timeout=FINISH_TIMEOUT_SECONDS)


async def test_a_clone_of_a_deleted_session_creates_nothing(
    db_session: AsyncSession,
) -> None:
    graph = await _seed_source(db_session)
    await db_session.execute(
        update(models.Session)
        .where(models.Session.id == graph.session.id)
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.commit()

    with pytest.raises(ResourceNotFoundException):
        await crud.clone_session(db_session, graph.workspace.name, graph.session.name)

    assert set(await _sessions(db_session, graph)) == {graph.session.name}
