"""Enqueueing runs behind the lifecycle barrier.

The message routes commit their messages and enqueue the work for them in a
background task. That task used to get-or-create the session, so a session
deleted in between came back empty, with work queued for messages that were
gone. A delayed dream sits in a timer in one process, where a deletion handled
by another replica cannot cancel it.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.lifecycle_barrier import capture_write_snapshot
from src.deriver.enqueue import enqueue, enqueue_dream
from src.dreamer.dream_scheduler import DreamScheduler
from tests.deriver.conftest import independent_sessions

enqueue_module = importlib.import_module("src.deriver.enqueue")

InsideTheEnqueue = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Written:
    workspace: str
    session: str
    session_id: str
    peer: str
    payloads: list[dict[str, Any]]


async def _write_messages(db: AsyncSession, count: int = 2) -> Written:
    """What a message route leaves behind: committed messages and their payloads."""
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.commit()
    session_name = str(generate_nanoid())
    messages = await crud.create_messages(
        db,
        [
            schemas.MessageCreate(content=f"message {index}", peer_id="alice")
            for index in range(count)
        ],
        workspace.name,
        session_name,
    )
    session_id = await db.scalar(
        select(models.Session.id).where(
            models.Session.workspace_name == workspace.name,
            models.Session.name == session_name,
        )
    )
    assert isinstance(session_id, str)
    written = Written(
        workspace=workspace.name,
        session=session_name,
        session_id=session_id,
        peer="alice",
        payloads=[
            {
                "workspace_name": workspace.name,
                "session_name": session_name,
                "message_id": message.id,
                "content": message.content,
                "peer_name": message.peer_name,
                "created_at": message.created_at,
                "message_public_id": message.public_id,
                "seq_in_session": message.seq_in_session,
                "configuration": None,
            }
            for message in messages
        ],
    )
    await db.commit()
    return written


async def _queued(
    db: AsyncSession, written: Written
) -> list[tuple[int | None, str | None]]:
    """``(message_id, session_id)`` of each representation row queued for the workspace."""
    rows = await db.execute(
        select(models.QueueItem.message_id, models.QueueItem.session_id)
        .where(
            models.QueueItem.workspace_name == written.workspace,
            models.QueueItem.task_type == "representation",
        )
        .order_by(models.QueueItem.message_id)
    )
    return [(row.message_id, row.session_id) for row in rows]


async def _count(db: AsyncSession, model: Any, *where: Any) -> int:
    return int(
        await db.scalar(select(func.count()).select_from(model).where(*where)) or 0
    )


async def test_new_messages_are_queued_under_their_sessions_id(
    db_session: AsyncSession,
) -> None:
    written = await _write_messages(db_session)

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == [
        (payload["message_id"], written.session_id) for payload in written.payloads
    ]


async def test_a_session_deleted_before_the_enqueue_is_not_recreated(
    db_session: AsyncSession,
) -> None:
    written = await _write_messages(db_session)
    await crud.delete_session(db_session, written.workspace, written.session)

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == []
    assert (
        await _count(
            db_session,
            models.Session,
            models.Session.workspace_name == written.workspace,
        )
        == 0
    )


async def test_a_workspace_deleted_before_the_enqueue_is_not_recreated(
    db_session: AsyncSession,
) -> None:
    written = await _write_messages(db_session)
    await crud.delete_session(db_session, written.workspace, written.session)
    await crud.delete_workspace(db_session, written.workspace)

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == []
    assert (
        await _count(
            db_session, models.Workspace, models.Workspace.name == written.workspace
        )
        == 0
    )


async def test_a_session_recreated_under_the_same_name_gets_no_work(
    db_session: AsyncSession,
) -> None:
    """The messages belonged to the old session and went with it."""
    written = await _write_messages(db_session)
    await crud.delete_session(db_session, written.workspace, written.session)
    db_session.add(
        models.Session(name=written.session, workspace_name=written.workspace)
    )
    await db_session.commit()

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == []


async def test_a_deleted_message_gets_no_work(db_session: AsyncSession) -> None:
    written = await _write_messages(db_session)
    deleted, kept = written.payloads
    await db_session.execute(
        update(models.Message)
        .where(
            models.Message.tenant_id == models.DEFAULT_TENANT_ID,
            models.Message.id == deleted["message_id"],
        )
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.commit()

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == [
        (kept["message_id"], written.session_id)
    ]


async def test_a_session_deleted_inside_the_enqueue_gets_no_work(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deletion lands after the enqueue read the session, before its insert."""
    written = await _write_messages(db_session)
    handle_session = enqueue_module.handle_session

    async def delete_then_handle(*args: Any, **kwargs: Any) -> Any:
        records = await handle_session(*args, **kwargs)
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, written.workspace, written.session)
        return records

    monkeypatch.setattr(enqueue_module, "handle_session", delete_then_handle)

    await enqueue(written.payloads, tenant_id=None)

    assert await _queued(db_session, written) == []
    assert (
        await _count(
            db_session,
            models.Session,
            models.Session.workspace_name == written.workspace,
        )
        == 0
    )


async def _seed_representation(db: AsyncSession) -> tuple[str, str, str]:
    """A workspace, a peer, and one explicit conclusion in the peer's own collection."""
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.flush()
    peer = models.Peer(name="alice", workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add_all([peer, session])
    await db.flush()
    db.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db.flush()
    db.add(
        models.Document(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            session_name=session.name,
            content="alice drinks tea",
            level="explicit",
        )
    )
    await db.commit()
    return workspace.name, peer.name, session.name


async def _queued_dreams(db: AsyncSession, workspace_name: str) -> int:
    return await _count(
        db,
        models.QueueItem,
        models.QueueItem.workspace_name == workspace_name,
        models.QueueItem.task_type == "dream",
    )


async def test_a_dream_is_queued_for_live_peers(db_session: AsyncSession) -> None:
    workspace, peer, _ = await _seed_representation(db_session)

    await enqueue_dream(
        workspace, observer=peer, observed=peer, dream_type=schemas.DreamType.OMNI
    )

    assert await _queued_dreams(db_session, workspace) == 1


async def test_a_dream_is_not_queued_for_a_deleted_peer(
    db_session: AsyncSession,
) -> None:
    workspace, peer, _ = await _seed_representation(db_session)
    await db_session.execute(
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace, models.Peer.name == peer)
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.commit()

    await enqueue_dream(
        workspace, observer=peer, observed=peer, dream_type=schemas.DreamType.OMNI
    )

    assert await _queued_dreams(db_session, workspace) == 0


async def test_a_delayed_dream_is_not_queued_after_another_replica_deleted_its_peer(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timer was set here. The deletion ran elsewhere and could not cancel it."""
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    workspace, peer, _ = await _seed_representation(db_session)
    # What check_and_schedule_dream captures when it sets the timer.
    identity = await capture_write_snapshot(db_session, workspace, peer_names=[peer])
    await db_session.commit()

    # Another replica deletes the peer.
    async with independent_sessions(db_engine)() as other_replica:
        await other_replica.execute(
            update(models.Peer)
            .where(models.Peer.workspace_name == workspace, models.Peer.name == peer)
            .values(deleted_at=datetime.now(UTC))
        )
        await other_replica.commit()

    # The timer fires.
    await DreamScheduler().execute_dream(
        workspace,
        schemas.DreamType.OMNI,
        observer=peer,
        observed=peer,
        barrier=identity,
    )

    assert await _queued_dreams(db_session, workspace) == 0


async def test_a_delayed_dream_is_not_queued_for_a_peer_recreated_under_the_same_name(
    db_session: AsyncSession,
) -> None:
    """The timer holds the peer's id, so a new peer with the old name is another peer."""
    workspace, peer, session = await _seed_representation(db_session)
    identity = await capture_write_snapshot(db_session, workspace, peer_names=[peer])
    await db_session.commit()

    await crud.delete_session(db_session, workspace, session)
    for model, column in (
        (models.Collection, models.Collection.observer),
        (models.Peer, models.Peer.name),
    ):
        await db_session.execute(
            delete(model).where(model.workspace_name == workspace, column == peer)
        )
    db_session.add(models.Peer(name=peer, workspace_name=workspace))
    await db_session.commit()

    await enqueue_dream(
        workspace,
        observer=peer,
        observed=peer,
        dream_type=schemas.DreamType.OMNI,
        barrier=identity,
    )
    assert await _queued_dreams(db_session, workspace) == 0

    # Decided now, for the new peer, it is queued.
    await enqueue_dream(
        workspace, observer=peer, observed=peer, dream_type=schemas.DreamType.OMNI
    )
    assert await _queued_dreams(db_session, workspace) == 1
