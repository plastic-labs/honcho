"""The per-collection lock that keeps concurrent document creation from duplicating."""

import asyncio

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.config import settings
from src.crud.document import (
    _collection_documents_lock_key,  # pyright: ignore[reportPrivateUsage]
    _lock_collection_documents,  # pyright: ignore[reportPrivateUsage]
)
from src.db import tenant_context


def test_key_is_stable_and_separates_collections():
    key = _collection_documents_lock_key("ws", observer="alice", observed="bob")

    assert key == _collection_documents_lock_key("ws", observer="alice", observed="bob")
    assert -(2**63) <= key < 2**63
    assert key != _collection_documents_lock_key("ws", observer="bob", observed="alice")
    assert key != _collection_documents_lock_key(
        "other", observer="alice", observed="bob"
    )
    # Parts that would run together under a plain separator stay apart.
    assert _collection_documents_lock_key(
        "ws", observer="a", observed="b:c"
    ) != _collection_documents_lock_key("ws", observer="a:b", observed="c")


def test_key_carries_the_tenant_under_multi_tenant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    keys: list[int] = []
    for tenant in ("tenant-a", "tenant-b"):
        token = tenant_context.set(tenant)
        try:
            keys.append(
                _collection_documents_lock_key(
                    "default", observer="alice", observed="alice"
                )
            )
        finally:
            tenant_context.reset(token)
    assert keys[0] != keys[1]

    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="no tenant bound"):
            _collection_documents_lock_key(
                "default", observer="alice", observed="alice"
            )
    finally:
        tenant_context.reset(token)


async def _setup_collection(
    db_session: AsyncSession, workspace: models.Workspace, observer: models.Peer
) -> tuple[str, str]:
    """Commit a session and a collection that independent connections can see."""
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([observed, session])
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
        )
    )
    await db_session.commit()
    return observed.name, session.name


def _document(content: str, session_name: str) -> schemas.DocumentCreate:
    return schemas.DocumentCreate(
        content=content,
        session_name=session_name,
        level="explicit",
        embedding=[0.5] * 1536,
        metadata=schemas.DocumentMetadata(
            message_ids=[1], message_created_at="2025-01-01T00:00:00Z"
        ),
    )


async def _live_document_count(
    db: AsyncSession, workspace_name: str, observer: str, observed: str
) -> int:
    return (
        await db.scalar(
            select(func.count())
            .select_from(models.Document)
            .where(
                models.Document.workspace_name == workspace_name,
                models.Document.observer == observer,
                models.Document.observed == observed,
                models.Document.deleted_at.is_(None),
            )
        )
    ) or 0


async def _advisory_waiters(db: AsyncSession) -> int:
    return (
        await db.scalar(
            text(
                "SELECT count(*) FROM pg_locks "
                + "WHERE locktype = 'advisory' AND NOT granted "
                + "AND database = (SELECT oid FROM pg_database "
                + "WHERE datname = current_database())"
            )
        )
    ) or 0


@pytest.mark.asyncio
async def test_second_writer_waits_then_sees_the_first_writers_document(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """The interleaving that used to duplicate: both writers start before either commits."""
    workspace, observer = sample_data
    observed, session_name = await _setup_collection(db_session, workspace, observer)
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def create(db: AsyncSession) -> crud.CreateDocumentsResult:
        return await crud.create_documents(
            db,
            [_document("Alice lives in Lisbon", session_name)],
            workspace.name,
            observer=observer.name,
            observed=observed,
        )

    async with Session() as first, Session() as second:
        # The first writer has started (it holds the lock) and has not committed.
        await _lock_collection_documents(
            first, workspace.name, observer=observer.name, observed=observed
        )

        second_task = asyncio.create_task(create(second))
        try:
            async with asyncio.timeout(10):
                while await _advisory_waiters(db_session) == 0:
                    await asyncio.sleep(0.02)
            assert not second_task.done()

            first_result = await create(first)
            second_result = await asyncio.wait_for(second_task, timeout=10)
        finally:
            second_task.cancel()

    assert len(first_result.created_documents) == 1
    assert second_result.created_documents == []
    assert second_result.exact_dup_existing_count == 1
    assert (
        await _live_document_count(db_session, workspace.name, observer.name, observed)
        == 1
    )


@pytest.mark.asyncio
async def test_parallel_writers_of_one_conclusion_save_it_once(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session_name = await _setup_collection(db_session, workspace, observer)
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def create() -> crud.CreateDocumentsResult:
        async with Session() as db:
            return await crud.create_documents(
                db,
                [_document("Alice lives in Lisbon", session_name)],
                workspace.name,
                observer=observer.name,
                observed=observed,
            )

    results = await asyncio.gather(*(create() for _ in range(6)))

    assert sum(len(result.created_documents) for result in results) == 1
    assert sum(result.exact_dup_existing_count for result in results) == 5
    assert (
        await _live_document_count(db_session, workspace.name, observer.name, observed)
        == 1
    )


@pytest.mark.asyncio
async def test_other_collections_do_not_wait(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, observer = sample_data
    observed, session_name = await _setup_collection(db_session, workspace, observer)
    Session = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async with Session() as holder, Session() as writer:
        # Hold the lock of a different collection: the reverse pair.
        await _lock_collection_documents(
            holder, workspace.name, observer=observed, observed=observer.name
        )
        result = await asyncio.wait_for(
            crud.create_documents(
                writer,
                [_document("Alice lives in Lisbon", session_name)],
                workspace.name,
                observer=observer.name,
                observed=observed,
            ),
            timeout=10,
        )
        await holder.rollback()

    assert len(result.created_documents) == 1
