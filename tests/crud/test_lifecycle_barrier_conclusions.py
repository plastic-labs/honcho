"""Public conclusion creation obeys parent liveness.

``create_observations`` looks its sessions and peers up, calls the embedding
API, and then inserts. The lookups are not a check: a deletion can land during
the embedding call. Each test lets one land there.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models, schemas
from src.embedding_client import embedding_client
from src.exceptions import WriteParentGoneError
from tests.deriver.conftest import independent_sessions
from tests.lifecycle_barrier_graph import Graph, document_count, seed_graph, tombstone

DuringEmbedding = Callable[[], Awaitable[None]]


def _conclusion(
    graph: Graph, content: str = "ann drinks coffee", session: str | None = None
) -> schemas.ConclusionCreate:
    return schemas.ConclusionCreate(
        content=content,
        observer_id=graph.observer.name,
        observed_id=graph.observed.name,
        session_id=session if session is not None else graph.session.name,
    )


async def _create(
    db: AsyncSession,
    graph: Graph,
    monkeypatch: pytest.MonkeyPatch,
    conclusions: list[schemas.ConclusionCreate],
    during_embedding: DuringEmbedding | None = None,
) -> list[models.Document]:
    """Create conclusions; a deletion can land while they are being embedded."""

    async def embed(texts: list[str], **_kwargs: Any) -> list[list[float]]:
        if during_embedding is not None:
            await during_embedding()
        return [[0.1] * 1536 for _ in texts]

    monkeypatch.setattr(embedding_client, "simple_batch_embed", embed)
    return await crud.create_observations(db, conclusions, graph.workspace.name)


async def _collections(db: AsyncSession, graph: Graph) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(models.Collection)
            .where(models.Collection.workspace_name == graph.workspace.name)
        )
        or 0
    )


async def test_conclusions_are_created_with_their_collection(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = await seed_graph(db_session, with_collection=False)

    created = await _create(db_session, graph, monkeypatch, [_conclusion(graph)])

    assert [doc.content for doc in created] == ["ann drinks coffee"]
    assert await _collections(db_session, graph) == 1
    explicit_count = await db_session.scalar(
        select(models.Collection.explicit_count).where(
            models.Collection.workspace_name == graph.workspace.name
        )
    )
    assert explicit_count == 1


async def test_a_session_deleted_during_the_request_gets_no_conclusion(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = await seed_graph(db_session, with_collection=False)

    async def delete_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, graph.workspace.name, graph.session.name)

    with pytest.raises(WriteParentGoneError) as raised:
        await _create(
            db_session, graph, monkeypatch, [_conclusion(graph)], delete_the_session
        )

    assert raised.value.scope == "session"
    assert raised.value.status_code == 404
    assert await document_count(db_session, graph) == 0
    # No collection was left behind, and the session was not recreated.
    assert await _collections(db_session, graph) == 0
    assert (
        await db_session.scalar(
            select(models.Session.id).where(
                models.Session.workspace_name == graph.workspace.name
            )
        )
        is None
    )


@pytest.mark.parametrize("kind", ["workspace", "observer", "observed"])
async def test_a_parent_deleted_during_the_request_gets_no_conclusion(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    graph = await seed_graph(db_session, with_collection=False)

    async def delete_the_parent() -> None:
        async with independent_sessions(db_engine)() as db:
            await tombstone(db, graph, kind)

    with pytest.raises(WriteParentGoneError):
        await _create(
            db_session, graph, monkeypatch, [_conclusion(graph)], delete_the_parent
        )

    assert await document_count(db_session, graph) == 0
    assert await _collections(db_session, graph) == 0


async def test_one_deleted_session_refuses_the_whole_request(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request can name several sessions. It is written whole or not at all."""
    graph = await seed_graph(db_session)
    db_session.add(models.Session(name="other", workspace_name=graph.workspace.name))
    await db_session.commit()
    conclusions = [
        _conclusion(graph, "ann drinks coffee"),
        _conclusion(graph, "ann works nights", session="other"),
    ]

    created = await _create(db_session, graph, monkeypatch, conclusions)
    assert len(created) == 2

    async def delete_the_other_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, graph.workspace.name, "other")

    with pytest.raises(WriteParentGoneError):
        await _create(
            db_session,
            graph,
            monkeypatch,
            [
                _conclusion(graph, "ann likes jazz"),
                _conclusion(graph, "ann owns a cat", session="other"),
            ],
            delete_the_other_session,
        )

    contents = set(
        await db_session.scalars(
            select(models.Document.content).where(
                models.Document.workspace_name == graph.workspace.name
            )
        )
    )
    assert contents == {"ann drinks coffee"}
