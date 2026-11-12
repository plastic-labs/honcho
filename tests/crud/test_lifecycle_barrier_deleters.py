"""Each deletion path keeps the deleter's side of the lifecycle barrier.

A producer captures its snapshot, the deletion commits, and the producer's
final write must then be refused. Without the deleter's lock and generation
bump the write would land under the deleted parent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud.lifecycle_barrier import final_write
from src.exceptions import WriteGenerationStaleError, WriteParentGoneError
from tests.lifecycle_barrier_graph import (
    Graph,
    capture,
    document,
    document_count,
    seed_graph,
    write,
)


async def _collection_generation(db: AsyncSession, graph: Graph) -> int:
    generation = await db.scalar(
        select(models.Collection.deletion_generation).where(
            models.Collection.tenant_id == graph.tenant_id,
            models.Collection.id == graph.collection.id,
        )
    )
    assert isinstance(generation, int)
    return generation


async def _seed_conclusion(db: AsyncSession, graph: Graph) -> str:
    conclusion = document(graph, "a conclusion from the session")
    db.add(conclusion)
    await db.commit()
    return conclusion.id


async def test_session_deletion_refuses_a_write_into_the_session(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    await crud.delete_session(db_session, graph.workspace.name, graph.session.name)

    with pytest.raises(WriteParentGoneError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "session"
    assert await document_count(db_session, graph) == 0


async def test_session_deletion_is_a_deletion_in_each_representation_it_empties(
    db_session: AsyncSession,
) -> None:
    """A dream reads a collection across sessions and names no session.

    Only the collection's generation tells it that conclusions it read are gone.
    """
    graph = await seed_graph(db_session)
    await _seed_conclusion(db_session, graph)
    # Captured as a dream captures: the representation, no session, no messages.
    snapshot = await capture(db_session, graph, session_name=None, message_ids=[])

    await crud.delete_session(db_session, graph.workspace.name, graph.session.name)
    assert await _collection_generation(db_session, graph) == 1

    # The dream's write: a deduction with no session, resting on what it read.
    deduction = document(graph, "a deduction from the erased conclusion")
    deduction.session_name = None
    deduction.level = "deductive"
    with pytest.raises(WriteGenerationStaleError) as raised:
        async with final_write(db_session, snapshot):
            db_session.add(deduction)
    await db_session.rollback()
    assert raised.value.scope == "collection"
    assert await document_count(db_session, graph) == 0


async def test_session_deletion_leaves_untouched_representations_alone(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    await crud.delete_session(db_session, graph.workspace.name, graph.session.name)
    assert await _collection_generation(db_session, graph) == 0


async def test_workspace_deletion_refuses_a_write_into_the_workspace(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    snapshot = await capture(db_session, graph)

    await crud.delete_workspace(db_session, graph.workspace.name)

    with pytest.raises(WriteParentGoneError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "workspace"


@pytest.mark.parametrize("route", ["by_id", "by_collection"])
async def test_conclusion_deletion_is_a_deletion_in_its_representation(
    db_session: AsyncSession, route: str
) -> None:
    """A write computed while the erased conclusion was live must not land."""
    graph = await seed_graph(db_session)
    conclusion_id = await _seed_conclusion(db_session, graph)
    snapshot = await capture(db_session, graph)

    if route == "by_id":
        await crud.delete_document_by_id(
            db_session, graph.workspace.name, conclusion_id
        )
    else:
        await crud.delete_document(
            db_session,
            graph.workspace.name,
            conclusion_id,
            observer=graph.observer.name,
            observed=graph.observed.name,
        )
    assert await _collection_generation(db_session, graph) == 1

    with pytest.raises(WriteGenerationStaleError) as raised:
        await write(db_session, graph, snapshot)
    assert raised.value.scope == "collection"


async def test_a_dreams_own_removal_is_not_a_deletion(
    db_session: AsyncSession,
) -> None:
    """Consolidation removes a conclusion without erasing its sources.

    The dream that removed it, and a deriver writing beside it, keep working.
    """
    graph = await seed_graph(db_session)
    conclusion_id = await _seed_conclusion(db_session, graph)
    snapshot = await capture(db_session, graph)

    removed = await crud.delete_documents(
        db_session,
        graph.workspace.name,
        [conclusion_id],
        observer=graph.observer.name,
        observed=graph.observed.name,
    )
    assert [doc_id for doc_id, _ in removed] == [conclusion_id]
    assert await _collection_generation(db_session, graph) == 0

    await write(db_session, graph, snapshot)
