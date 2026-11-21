"""Write helpers stage their change in the caller's transaction and never commit.

A change and the record that reports it have to commit together, so the helpers
on the write path leave the commit to their caller. These tests pin that: a
source guard that no listed helper calls ``commit()``, and per helper, that its
write is invisible to other connections until the caller commits, with the
deferred cache or vector work running only from ``post_commit()``.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import textwrap
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.crud import collection as collection_crud
from src.crud.document import create_documents
from src.crud.peer_card import construct_peer_card_label, set_peer_card
from src.deriver.enqueue import (
    _insert_queue_records,  # pyright: ignore[reportPrivateUsage]
)
from src.exceptions import ResourceNotFoundException
from src.routers.messages import create_messages_with_file
from src.utils.summarizer import (
    SUMMARIES_KEY,
    Summary,
    SummaryType,
    _save_summary,  # pyright: ignore[reportPrivateUsage]
)

# Every helper that stages a change for its caller to commit.
CALLER_COMMITS: list[Callable[..., Any]] = [
    create_documents,
    collection_crud.get_or_create_collection,
    set_peer_card,
    _save_summary,
    _insert_queue_records,
    create_messages_with_file,
]


def _commit_calls(source: str) -> list[int]:
    """Line numbers of every ``<x>.commit(...)`` call in ``source``."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "commit"
    ]


@pytest.mark.parametrize("helper", CALLER_COMMITS, ids=lambda h: h.__qualname__)
def test_write_helpers_never_commit(helper: Callable[..., Any]) -> None:
    """The caller owns the transaction; a commit inside would split it."""
    assert _commit_calls(inspect.getsource(inspect.unwrap(helper))) == []


def test_the_guard_sees_a_commit() -> None:
    """Without this, an empty result could mean the guard checks nothing."""
    source = """
        async def helper(db):
            db.add(row)
            async def on_commit():
                await db.commit()
            await db.commit()
    """
    assert len(_commit_calls(source)) == 2


def _other_sessions(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


async def _committed_session(
    db_session: AsyncSession, workspace: models.Workspace
) -> models.Session:
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()
    return session


@pytest.mark.asyncio
async def test_set_peer_card_waits_for_the_caller_commit(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data
    label = construct_peer_card_label(observer=peer.name, observed=peer.name)

    async def committed_card() -> Any:
        async with _other_sessions(db_engine)() as other:
            return await other.scalar(
                select(models.Peer.internal_metadata[label]).where(
                    models.Peer.workspace_name == workspace.name,
                    models.Peer.name == peer.name,
                )
            )

    write = await set_peer_card(
        db_session,
        workspace.name,
        ["likes tea"],
        observer=peer.name,
        observed=peer.name,
    )
    assert await committed_card() is None

    await db_session.commit()
    await write.post_commit()
    assert await committed_card() == ["likes tea"]


@pytest.mark.asyncio
async def test_save_summary_waits_for_the_caller_commit(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _peer = sample_data
    session = await _committed_session(db_session, workspace)
    summary: Summary = {
        "content": "they talked about tea",
        "message_id": 1,
        "summary_type": SummaryType.SHORT.value,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "token_count": 5,
        "message_public_id": "m1",
    }

    async def committed_summaries() -> Any:
        async with _other_sessions(db_engine)() as other:
            return await other.scalar(
                select(models.Session.internal_metadata[SUMMARIES_KEY]).where(
                    models.Session.workspace_name == workspace.name,
                    models.Session.name == session.name,
                )
            )

    write = await _save_summary(db_session, summary, workspace.name, session.name)
    assert await committed_summaries() is None

    await db_session.commit()
    await write.post_commit()
    stored = await committed_summaries()
    assert stored[SummaryType.SHORT.value]["content"] == "they talked about tea"


@pytest.mark.asyncio
async def test_create_documents_waits_for_the_caller_commit(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """The rows commit with the caller; their vectors sync only after that."""
    workspace, peer = sample_data
    session = await _committed_session(db_session, workspace)
    created = await collection_crud.get_or_create_collection(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    await db_session.commit()
    await created.post_commit()

    result = await create_documents(
        db_session,
        [
            schemas.DocumentCreate(
                content="likes tea",
                embedding=[0.5] * 1536,
                session_name=session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1], message_created_at="2025-01-01T00:00:00Z"
                ),
            )
        ],
        workspace.name,
        observer=peer.name,
        observed=peer.name,
    )
    assert len(result.created_documents) == 1

    async def committed_sync_states() -> list[str]:
        async with _other_sessions(db_engine)() as other:
            rows = await other.scalars(
                select(models.Document.sync_state).where(
                    models.Document.workspace_name == workspace.name,
                    models.Document.observer == peer.name,
                )
            )
            return list(rows)

    assert await committed_sync_states() == []

    await db_session.commit()
    assert await committed_sync_states() == ["pending"]

    await result.post_commit()
    assert await committed_sync_states() == ["synced"]


@pytest.mark.asyncio
async def test_get_or_create_collection_waits_for_the_caller_commit(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data

    async def committed_count() -> int:
        async with _other_sessions(db_engine)() as other:
            return (
                await other.scalar(
                    select(func.count())
                    .select_from(models.Collection)
                    .where(models.Collection.workspace_name == workspace.name)
                )
                or 0
            )

    result = await collection_crud.get_or_create_collection(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert result.created
    assert await committed_count() == 0

    await db_session.commit()
    await result.post_commit()
    assert await committed_count() == 1

    again = await collection_crud.get_or_create_collection(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert not again.created
    assert again.resource.id == result.resource.id


@pytest.mark.asyncio
async def test_a_lost_collection_create_race_keeps_the_caller_work(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """The insert's savepoint rolls back alone, so earlier staged writes survive."""
    workspace, peer = sample_data
    existing = models.Collection(
        workspace_name=workspace.name, observer=peer.name, observed=peer.name
    )
    db_session.add(existing)
    await db_session.commit()

    # Staged before the collection call, in the same transaction.
    staged_peer = models.Peer(
        name=str(generate_nanoid()), workspace_name=workspace.name
    )
    db_session.add(staged_peer)
    await db_session.flush()

    real_get_collection = collection_crud.get_collection
    calls = 0

    async def miss_once(*args: Any, **kwargs: Any) -> models.Collection:
        # The first lookup misses as if a concurrent writer committed the row
        # just after it, so the insert hits the unique constraint.
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResourceNotFoundException("not yet")
        return await real_get_collection(*args, **kwargs)

    with patch.object(collection_crud, "get_collection", miss_once):
        result = await collection_crud.get_or_create_collection(
            db_session, workspace.name, observer=peer.name, observed=peer.name
        )

    assert not result.created
    assert result.resource.id == existing.id
    await db_session.commit()
    assert (
        await db_session.scalar(
            select(models.Peer.name).where(
                models.Peer.workspace_name == workspace.name,
                models.Peer.name == staged_peer.name,
            )
        )
        == staged_peer.name
    )


@pytest.mark.asyncio
async def test_creating_a_collection_keeps_the_caller_row_locks(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """Scope backfill locks membership, then creates collections: the lock must hold."""
    workspace, peer = sample_data
    session = await _committed_session(db_session, workspace)
    db_session.add(
        models.SessionPeer(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
        )
    )
    await db_session.commit()

    await db_session.execute(
        select(models.SessionPeer.peer_name)
        .where(
            models.SessionPeer.workspace_name == workspace.name,
            models.SessionPeer.session_name == session.name,
        )
        .with_for_update()
    )
    result = await collection_crud.get_or_create_collection(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert result.created

    async with _other_sessions(db_engine)() as other:
        with pytest.raises(DBAPIError, match="could not obtain lock"):
            await other.execute(
                select(models.SessionPeer.peer_name)
                .where(
                    models.SessionPeer.workspace_name == workspace.name,
                    models.SessionPeer.session_name == session.name,
                )
                .with_for_update(nowait=True)
            )
        await other.rollback()

    await db_session.commit()
    await result.post_commit()


@pytest.mark.asyncio
async def test_upload_metadata_is_written_with_the_message(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    """One insert carries the file metadata; no second write follows it."""
    workspace, peer = sample_data
    session_name = str(generate_nanoid())
    created = await crud.create_messages(
        db_session,
        messages=[
            schemas.MessageCreate(content="chunk one", peer_id=peer.name),
            schemas.MessageCreate(content="chunk two", peer_id=peer.name),
        ],
        workspace_name=workspace.name,
        session_name=session_name,
        internal_metadata=[{"chunk_index": 0}, {"chunk_index": 1}],
    )
    assert [m.internal_metadata for m in created] == [
        {"chunk_index": 0},
        {"chunk_index": 1},
    ]
    stored = (
        await db_session.execute(
            text(
                "SELECT internal_metadata FROM messages "
                + "WHERE workspace_name = :w AND session_name = :s "
                + "ORDER BY seq_in_session"
            ),
            {"w": workspace.name, "s": session_name},
        )
    ).scalars()
    assert list(stored) == [{"chunk_index": 0}, {"chunk_index": 1}]


@pytest.mark.asyncio
async def test_create_messages_rejects_mismatched_internal_metadata(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data
    with pytest.raises(ValueError, match="one entry per message"):
        await crud.create_messages(
            db_session,
            messages=[schemas.MessageCreate(content="only", peer_id=peer.name)],
            workspace_name=workspace.name,
            session_name=str(generate_nanoid()),
            internal_metadata=[{}, {}],
        )
