"""Session and workspace deletion take dream history in the same lock order."""

import asyncio
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud import session as session_crud
from src.crud.dream_history import delete_session_history, delete_workspace_history
from src.db import SessionLocal
from src.models import Peer, Workspace


async def test_session_and_workspace_deletion_do_not_deadlock(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """A session deletion paused while it holds its history rows, and a workspace
    deletion started meanwhile, both finish: the workspace waits instead of
    taking rows the session deletion needs next."""
    workspace, peer = sample_data
    workspace_name = workspace.name
    session_name = str(generate_nanoid())
    db_session.add(models.Session(name=session_name, workspace_name=workspace_name))
    await db_session.flush()
    db_session.add(
        models.Message(
            session_name=session_name,
            workspace_name=workspace_name,
            peer_name=peer.name,
            content="I switched to decaf",
            seq_in_session=1,
        )
    )
    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=peer.name,
        observed=peer.name,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    # A removed conclusion's copy from this session; its documents are gone.
    db_session.add(
        models.DreamRunChange(
            run_id=target.run_id,
            workspace_name=workspace_name,
            kind="removed",
            conclusion_id=generate_nanoid(),
            level="explicit",
            content="Ann drinks coffee daily",
            session_name=session_name,
        )
    )
    await db_session.commit()

    held = asyncio.Event()
    release = asyncio.Event()
    original = delete_session_history

    async def pause_after_history(*args: Any, **kwargs: Any) -> None:
        await original(*args, **kwargs)
        held.set()
        await release.wait()

    monkeypatch.setattr(session_crud, "delete_session_history", pause_after_history)

    async def delete_session() -> None:
        async with SessionLocal() as db:
            await crud.delete_session(db, workspace_name, session_name)

    async def delete_workspace() -> None:
        async with SessionLocal() as db:
            await crud.delete_workspace(db, workspace_name)

    session_task = asyncio.create_task(delete_session())
    await asyncio.wait_for(held.wait(), timeout=10)
    workspace_task = asyncio.create_task(delete_workspace())
    # Longer than Postgres's default deadlock_timeout (1s), so a lock cycle
    # would already have been detected and one side aborted.
    await asyncio.sleep(1.5)
    release.set()

    results = await asyncio.wait_for(
        asyncio.gather(session_task, workspace_task, return_exceptions=True),
        timeout=30,
    )
    assert [r for r in results if isinstance(r, BaseException)] == []


async def test_workspace_history_deletion_takes_changes_in_id_order(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Session deletion locks a session's changes in id order. Holding the lower
    one while workspace history deletion starts, then taking the higher one,
    must not deadlock: the workspace side waits on the lower one first instead
    of reaching the higher one through a run's cascade."""
    workspace, peer = sample_data
    workspace_name = workspace.name
    runs: list[str] = []
    for _ in range(2):
        target = await crud.open_run(
            run_id=generate_nanoid(),
            workspace_name=workspace_name,
            observer=peer.name,
            observed=peer.name,
            dream_type="omni",
            trigger_reason="manual",
        )
        assert target is not None
        runs.append(target.run_id)
    # The lower change id goes to the second run and the higher to the first,
    # so the runs and their changes sort in opposite orders.
    change_ids: list[int] = []
    for run_id in sorted(runs, reverse=True):
        change = models.DreamRunChange(
            run_id=run_id,
            workspace_name=workspace_name,
            kind="removed",
            conclusion_id=generate_nanoid(),
            level="explicit",
            content="Ann drinks coffee daily",
        )
        db_session.add(change)
        await db_session.flush()
        change_ids.append(change.id)
    await db_session.commit()
    low, high = sorted(change_ids)

    async with SessionLocal() as holder:
        await holder.execute(
            select(models.DreamRunChange.id)
            .where(models.DreamRunChange.id == low)
            .with_for_update()
        )

        async def delete_history() -> None:
            async with SessionLocal() as db:
                await delete_workspace_history(db, workspace_name)
                await db.commit()

        workspace_task = asyncio.create_task(delete_history())
        await asyncio.sleep(0.5)
        await asyncio.wait_for(
            holder.execute(
                select(models.DreamRunChange.id)
                .where(models.DreamRunChange.id == high)
                .with_for_update()
            ),
            timeout=5,
        )
        await holder.commit()

    await asyncio.wait_for(workspace_task, timeout=10)
    remaining = await db_session.scalar(
        select(models.DreamRun.id).where(
            models.DreamRun.workspace_name == workspace_name
        )
    )
    assert remaining is None


async def test_erasing_a_conclusion_does_not_wait_on_its_dreams_run(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A running dream holds its run while it writes. Erasure must not need that
    lock, or a dream, an erasure and a session deletion can wait on each other."""
    workspace, peer = sample_data
    workspace_name = workspace.name
    target = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=peer.name,
        observed=peer.name,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    conclusion_id = generate_nanoid()
    db_session.add(
        models.DreamRunChange(
            run_id=target.run_id,
            workspace_name=workspace_name,
            kind="removed",
            conclusion_id=conclusion_id,
            level="explicit",
            content="Ann drinks coffee daily",
        )
    )
    await db_session.commit()

    async with SessionLocal() as dream:
        await dream.execute(
            select(models.DreamRun.id)
            .where(models.DreamRun.id == target.run_id)
            .with_for_update()
        )
        async with SessionLocal() as db:
            await asyncio.wait_for(
                crud.delete_document_by_id(db, workspace_name, conclusion_id),
                timeout=5,
            )
        await dream.rollback()
