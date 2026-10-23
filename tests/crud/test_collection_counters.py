import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas


async def _peer(db_session: AsyncSession, workspace_name: str) -> models.Peer:
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add(peer)
    await db_session.flush()
    return peer


async def _collection(
    db_session: AsyncSession, workspace_name: str, observer: str, observed: str
) -> models.Collection:
    collection = models.Collection(
        workspace_name=workspace_name, observer=observer, observed=observed
    )
    db_session.add(collection)
    await db_session.commit()
    return collection


async def _counters(
    db_session: AsyncSession, collection: models.Collection
) -> tuple[int, datetime.datetime | None]:
    row = (
        await db_session.execute(
            select(
                models.Collection.explicit_count, models.Collection.newest_explicit_at
            ).where(models.Collection.id == collection.id)
        )
    ).one()
    return row[0], row[1]


async def _explicit_rows(
    db_session: AsyncSession, collection: models.Collection
) -> tuple[int, datetime.datetime | None]:
    row = (
        await db_session.execute(
            select(
                func.count(models.Document.id), func.max(models.Document.created_at)
            ).where(
                models.Document.workspace_name == collection.workspace_name,
                models.Document.observer == collection.observer,
                models.Document.observed == collection.observed,
                models.Document.level == "explicit",
            )
        )
    ).one()
    return row[0], row[1]


def _doc(
    collection: models.Collection, level: str, created_at: datetime.datetime
) -> models.Document:
    return models.Document(
        content="c",
        level=level,
        workspace_name=collection.workspace_name,
        observer=collection.observer,
        observed=collection.observed,
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_record_explicit_documents_counts_per_collection(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    workspace, peer = sample_data
    other = await _peer(db_session, workspace.name)
    first = await _collection(db_session, workspace.name, peer.name, peer.name)
    second = await _collection(db_session, workspace.name, peer.name, other.name)
    old = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    new = datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC)

    docs = [
        _doc(first, "explicit", old),
        _doc(first, "explicit", new),
        _doc(first, "deductive", new + datetime.timedelta(days=1)),
        _doc(second, "explicit", old),
    ]
    db_session.add_all(docs)
    await crud.record_explicit_documents(db_session, docs)
    await db_session.commit()

    assert await _counters(db_session, first) == (2, new)
    assert await _counters(db_session, second) == (1, old)

    later = [_doc(second, "explicit", old - datetime.timedelta(days=1))]
    db_session.add_all(later)
    await crud.record_explicit_documents(db_session, later)
    await db_session.commit()
    assert await _counters(db_session, second) == (2, old)


@pytest.mark.asyncio
async def test_create_observations_keeps_counters_consistent(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    workspace, peer = sample_data
    other = await _peer(db_session, workspace.name)
    await db_session.commit()

    await crud.create_observations(
        db_session,
        [
            schemas.ConclusionCreate(
                content=f"fact {i}", observer_id=peer.name, observed_id=other.name
            )
            for i in range(3)
        ],
        workspace.name,
    )

    collection = await crud.get_collection(
        db_session, workspace.name, observer=peer.name, observed=other.name
    )
    assert await _counters(db_session, collection) == await _explicit_rows(
        db_session, collection
    )


@pytest.mark.asyncio
async def test_create_documents_keeps_counters_consistent(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
):
    workspace, peer = sample_data
    other = await _peer(db_session, workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    collection = await _collection(db_session, workspace.name, peer.name, other.name)

    await crud.create_documents(
        db_session,
        [
            schemas.DocumentCreate(
                content="explicit one",
                level="explicit",
                session_name=session.name,
                embedding=[0.1] * 1536,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2024-01-01T00:00:00Z"
                ),
            ),
            schemas.DocumentCreate(
                content="explicit two",
                level="explicit",
                session_name=session.name,
                embedding=[0.3] * 1536,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2024-01-01T00:00:00Z"
                ),
            ),
            schemas.DocumentCreate(
                content="deduced",
                level="deductive",
                embedding=[0.2] * 1536,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2024-01-01T00:00:00Z"
                ),
            ),
        ],
        workspace.name,
        observer=peer.name,
        observed=other.name,
    )

    count, newest = await _counters(db_session, collection)
    assert count == 2
    assert (count, newest) == await _explicit_rows(db_session, collection)
