"""``create_documents`` behind the lifecycle barrier: inserts, reinforcement, fallback.

The reinforcement fallback is the narrow case. ``create_documents`` reads the
live conclusion it will reinforce, then locks it. If the row is tombstoned in
between, it inserts the incoming conclusion instead of dropping it. That is
right when a dream consolidated the row away, and wrong when a deletion erased
it: the insert would write the erased conclusion back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models, schemas
from src.crud import document as document_crud
from src.exceptions import WriteGenerationStaleError, WriteParentGoneError
from tests.deriver.conftest import independent_sessions
from tests.lifecycle_barrier_graph import (
    Graph,
    capture,
    document,
    document_count,
    seed_graph,
    tombstone,
)

CONTENT = "the observed peer drinks green tea"


def _incoming(graph: Graph) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=CONTENT,
        embedding=[0.9] * 1536,
        session_name=graph.session.name,
        metadata=schemas.DocumentMetadata(
            message_ids=[graph.message_id],
            message_created_at="2025-01-01T00:00:00Z",
        ),
    )


async def _seed_existing(db: AsyncSession, graph: Graph) -> str:
    existing = document(graph, CONTENT)
    db.add(existing)
    await db.commit()
    return existing.id


async def _live(db: AsyncSession, graph: Graph) -> list[tuple[str, int]]:
    """``(id, times_derived)`` of each live conclusion with the test content."""
    rows = await db.execute(
        select(models.Document.id, models.Document.times_derived).where(
            models.Document.tenant_id == graph.tenant_id,
            models.Document.workspace_name == graph.workspace.name,
            models.Document.content == CONTENT,
            models.Document.deleted_at.is_(None),
        )
    )
    return [(row.id, row.times_derived) for row in rows]


async def _create(
    db: AsyncSession,
    graph: Graph,
    snapshot: Any,
    between_read_and_lock: Callable[[], Awaitable[None]] | None = None,
) -> crud.CreateDocumentsResult:
    """Run ``create_documents``; optionally act after its dedup read, before its lock."""
    apply_row_updates = document_crud._apply_document_row_updates  # pyright: ignore[reportPrivateUsage]

    async def act_then_apply(*args: Any, **kwargs: Any) -> Any:
        if between_read_and_lock is not None:
            await between_read_and_lock()
        return await apply_row_updates(*args, **kwargs)

    with patch.object(document_crud, "_apply_document_row_updates", act_then_apply):
        return await crud.create_documents(
            db,
            [_incoming(graph)],
            graph.workspace.name,
            observer=graph.observer.name,
            observed=graph.observed.name,
            barrier=snapshot,
        )


async def test_an_insert_creates_the_missing_collection_in_the_same_transaction(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session, with_collection=False)
    snapshot = await capture(db_session, graph)

    result = await _create(db_session, graph, snapshot)

    assert len(result.created_documents) == 1
    assert len(await _live(db_session, graph)) == 1
    explicit_count = await db_session.scalar(
        select(models.Collection.explicit_count).where(
            models.Collection.workspace_name == graph.workspace.name
        )
    )
    assert explicit_count == 1


@pytest.mark.parametrize("kind", ["workspace", "session", "observer", "observed"])
async def test_an_insert_under_a_deleted_parent_writes_nothing(
    db_session: AsyncSession, kind: str
) -> None:
    graph = await seed_graph(db_session, with_collection=False)
    snapshot = await capture(db_session, graph)
    await tombstone(db_session, graph, kind)

    with pytest.raises(WriteParentGoneError):
        await _create(db_session, graph, snapshot)
    # Rolled back: the session is left usable for the caller that owns it.
    assert not db_session.in_transaction()

    assert await document_count(db_session, graph) == 0
    # The collection was not created under the deleted parent either.
    assert (
        await db_session.scalar(
            select(models.Collection.id).where(
                models.Collection.workspace_name == graph.workspace.name
            )
        )
        is None
    )


async def test_a_reinforcement_raises_the_count_and_inserts_nothing(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    existing_id = await _seed_existing(db_session, graph)
    snapshot = await capture(db_session, graph)

    result = await _create(db_session, graph, snapshot)

    assert result.exact_dup_existing_count == 1
    assert await _live(db_session, graph) == [(existing_id, 2)]


async def test_a_reinforcement_target_erased_before_the_lock_is_not_written_back(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    graph = await seed_graph(db_session)
    existing_id = await _seed_existing(db_session, graph)
    snapshot = await capture(db_session, graph)

    async def erase() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_document_by_id(db, graph.workspace.name, existing_id)

    with pytest.raises(WriteGenerationStaleError) as raised:
        await _create(db_session, graph, snapshot, erase)

    assert raised.value.scope == "collection"
    assert await _live(db_session, graph) == []


async def test_a_reinforcement_target_consolidated_before_the_lock_is_inserted(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """A dream removed the row. Nothing was erased, so the fallback stands."""
    graph = await seed_graph(db_session)
    existing_id = await _seed_existing(db_session, graph)
    snapshot = await capture(db_session, graph)

    async def consolidate() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_documents(
                db,
                graph.workspace.name,
                [existing_id],
                observer=graph.observer.name,
                observed=graph.observed.name,
            )

    result = await _create(db_session, graph, snapshot, consolidate)

    assert len(result.created_documents) == 1
    live = await _live(db_session, graph)
    assert len(live) == 1
    assert live[0][0] != existing_id
