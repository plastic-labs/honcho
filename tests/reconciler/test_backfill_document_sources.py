"""The reconciler drains legacy JSONB source linkage into document_sources."""

from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.reconciler.backfill_document_sources import (
    drain_document_sources_batch,
    has_pending_document_sources,
)


async def _collection(db: AsyncSession) -> tuple[str, str]:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.commit()
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add(peer)
    await db.commit()
    db.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db.commit()
    return workspace.name, peer.name


def _doc(
    workspace_name: str,
    peer: str,
    *,
    legacy: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> models.Document:
    return models.Document(
        workspace_name=workspace_name,
        observer=peer,
        observed=peer,
        content=f"doc {generate_nanoid()}",
        level="deductive",
        legacy_source_ids=legacy,
        internal_metadata=metadata or {},
    )


async def _reload(db: AsyncSession, doc_id: str) -> models.Document:
    """Re-query so the selectin relationship loads (refresh would lazy-load)."""
    db.expire_all()
    return (
        await db.execute(select(models.Document).where(models.Document.id == doc_id))
    ).scalar_one()


async def _edges(db: AsyncSession, derived_id: str) -> list[str]:
    rows = await db.execute(
        select(models.DocumentSource.source_id)
        .where(models.DocumentSource.derived_id == derived_id)
        .order_by(models.DocumentSource.position)
    )
    return list(rows.scalars().all())


@pytest.mark.asyncio
async def test_drain_covers_every_legacy_location(db_session: AsyncSession) -> None:
    ws, peer = await _collection(db_session)
    a, b, c, d, e = (generate_nanoid() for _ in range(5))

    column = _doc(ws, peer, legacy=[a, b])
    meta = _doc(ws, peer, metadata={"source_ids": [c], "keep": True})
    premise = _doc(ws, peer, metadata={"premise_ids": [d]})
    garbage = _doc(ws, peer, legacy=[e, "1234", "2024-01-01T00:00:00", e])
    db_session.add_all([column, meta, premise, garbage])
    await db_session.commit()
    ids = [column.id, meta.id, premise.id, garbage.id]

    assert await has_pending_document_sources(db_session)

    drained = await drain_document_sources_batch(db_session)
    await db_session.commit()
    assert drained == 4

    assert await _edges(db_session, ids[0]) == [a, b]
    assert await _edges(db_session, ids[1]) == [c]
    assert await _edges(db_session, ids[2]) == [d]
    assert await _edges(db_session, ids[3]) == [e]  # malformed + dupes dropped

    for doc_id in ids:
        doc = await _reload(db_session, doc_id)
        assert doc.legacy_source_ids is None
        assert "source_ids" not in doc.internal_metadata
        assert "premise_ids" not in doc.internal_metadata
    assert (await _reload(db_session, ids[1])).internal_metadata == {"keep": True}

    assert not await has_pending_document_sources(db_session)


@pytest.mark.asyncio
async def test_drain_is_batched_in_id_order(db_session: AsyncSession) -> None:
    ws, peer = await _collection(db_session)
    docs = [_doc(ws, peer, legacy=[generate_nanoid()]) for _ in range(5)]
    db_session.add_all(docs)
    await db_session.commit()

    assert await drain_document_sources_batch(db_session, batch_size=2) == 2
    await db_session.commit()

    # Order comes from the database collation, not Python's bytewise sort.
    ids = list(
        (
            await db_session.execute(
                select(models.Document.id).order_by(models.Document.id)
            )
        ).scalars()
    )
    for doc_id in ids:
        doc = await _reload(db_session, doc_id)
        assert (doc.legacy_source_ids is None) == (doc_id in ids[:2])

    assert await drain_document_sources_batch(db_session, batch_size=10) == 3
    await db_session.commit()
    assert await drain_document_sources_batch(db_session) == 0


@pytest.mark.asyncio
async def test_rows_already_on_the_edge_table_are_untouched(
    db_session: AsyncSession,
) -> None:
    ws, peer = await _collection(db_session)
    parent = generate_nanoid()
    linked = _doc(ws, peer)
    linked.source_ids = [parent]
    db_session.add(linked)
    await db_session.commit()

    assert not await has_pending_document_sources(db_session)
    assert await drain_document_sources_batch(db_session) == 0
    assert await _edges(db_session, linked.id) == [parent]


@pytest.mark.asyncio
async def test_source_ids_property_falls_back_until_drained(
    db_session: AsyncSession,
) -> None:
    ws, peer = await _collection(db_session)
    a, b = generate_nanoid(), generate_nanoid()
    doc = _doc(ws, peer, legacy=[a, "not-an-id", a, b])
    db_session.add(doc)
    await db_session.commit()
    doc_id = doc.id

    assert (await _reload(db_session, doc_id)).source_ids == [a, b]

    await drain_document_sources_batch(db_session)
    await db_session.commit()
    doc = await _reload(db_session, doc_id)

    assert doc.legacy_source_ids is None
    assert doc.source_ids == [a, b]
