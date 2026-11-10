"""A seeded workspace graph and a hand-played producer, for lifecycle barrier tests.

``seed_graph`` commits one workspace with an observer, an observed peer, a
session, one message and (by default) the observer's collection. ``capture``
and ``write`` are the two halves of a producer: the snapshot taken before the
input is read, and the final write behind the barrier.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.lifecycle_barrier import (
    WriteSnapshot,
    capture_write_snapshot,
    final_write,
)
from src.exceptions import LifecycleBarrierError

# Bounds on a hang, not timings.
LOCK_WAIT_TIMEOUT_SECONDS = 10.0
FINISH_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class Row:
    """Plain values of a seeded row, safe to read after a rollback."""

    id: str
    name: str


@dataclass(frozen=True)
class Graph:
    tenant_id: str
    workspace: Row
    observer: Row
    observed: Row
    session: Row
    collection: Row
    message_id: int
    message_public_id: str


MODELS: dict[str, Any] = {
    "workspace": models.Workspace,
    "session": models.Session,
    "observer": models.Peer,
    "observed": models.Peer,
}


async def seed_graph(db: AsyncSession, *, with_collection: bool = True) -> Graph:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.flush()
    observer = models.Peer(name="observer", workspace_name=workspace.name)
    observed = models.Peer(name="observed", workspace_name=workspace.name)
    session = models.Session(name="session", workspace_name=workspace.name)
    db.add_all([observer, observed, session])
    await db.flush()
    message = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=observed.name,
        content="a source message",
        seq_in_session=1,
    )
    collection = models.Collection(
        workspace_name=workspace.name, observer=observer.name, observed=observed.name
    )
    db.add(message)
    if with_collection:
        db.add(collection)
    await db.flush()
    graph = Graph(
        tenant_id=workspace.tenant_id,
        workspace=Row(workspace.id, workspace.name),
        observer=Row(observer.id, observer.name),
        observed=Row(observed.id, observed.name),
        session=Row(session.id, session.name),
        collection=Row(collection.id if with_collection else "", "collection"),
        message_id=message.id,
        message_public_id=message.public_id,
    )
    await db.commit()
    return graph


async def tombstone(db: AsyncSession, graph: Graph, kind: str) -> None:
    """What a tombstoning deletion leaves behind: ``deleted_at`` on the row."""
    model = MODELS[kind]
    await db.execute(
        update(model)
        .where(model.id == getattr(graph, kind).id)
        .values(deleted_at=datetime.now(UTC))
    )
    await db.commit()


async def capture(db: AsyncSession, graph: Graph, **overrides: Any) -> WriteSnapshot:
    kwargs: dict[str, Any] = {
        "session_name": graph.session.name,
        "representations": [(graph.observer.name, graph.observed.name)],
        "message_ids": [graph.message_public_id],
    }
    kwargs.update(overrides)
    snapshot = await capture_write_snapshot(db, graph.workspace.name, **kwargs)
    # A producer holds no transaction between its capture and its write.
    await db.commit()
    return snapshot


def document(graph: Graph, content: str = "a conclusion") -> models.Document:
    return models.Document(
        workspace_name=graph.workspace.name,
        observer=graph.observer.name,
        observed=graph.observed.name,
        session_name=graph.session.name,
        content=content,
        level="explicit",
    )


async def write(db: AsyncSession, graph: Graph, snapshot: WriteSnapshot) -> None:
    """The producer's final write: one document, behind the barrier."""
    try:
        async with final_write(db, snapshot):
            db.add(document(graph))
        await db.commit()
    except LifecycleBarrierError:
        await db.rollback()
        raise


async def document_count(db: AsyncSession, graph: Graph) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(models.Document)
            .where(
                models.Document.tenant_id == graph.tenant_id,
                models.Document.workspace_name == graph.workspace.name,
            )
        )
        or 0
    )


async def wait_until_waiting_on_a_lock(db: AsyncSession, pid: int) -> None:
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
