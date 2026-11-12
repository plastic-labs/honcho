"""Scope backfill writes its copies behind the lifecycle barrier.

A backfill plans from the session's explicit conclusions, embeds, and then
writes copies into the scope's collections, restoring a copy an earlier removal
soft-deleted. Each test lets a deletion land between the plan and the write.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.deriver import scope_backfill
from src.deriver.scope_backfill import (
    COPIED_FROM_KEY,
    process_scope_backfill,
    process_scope_removal,
)
from src.exceptions import WriteGenerationStaleError, WriteParentGoneError
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import scope_peer_name
from tests.deriver.conftest import independent_sessions

BetweenPlanAndWrite = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Seeded:
    workspace: str
    session: str
    sender: str
    scope_peer: str
    source_ids: list[str]


async def _seed(db: AsyncSession) -> Seeded:
    """A session in a scope, with two explicit conclusions about its sender.

    The scope's own collection does not exist yet: the backfill creates it.
    """
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.flush()
    sender = models.Peer(name="alice", workspace_name=workspace.name)
    scope = models.Peer(
        name=scope_peer_name(str(generate_nanoid())),
        workspace_name=workspace.name,
        internal_metadata={"kind": "scope"},
        configuration={"observe_me": False},
    )
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add_all([sender, scope, session])
    await db.flush()
    db.add(
        models.SessionPeer(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=scope.name,
        )
    )
    db.add(
        models.Collection(
            workspace_name=workspace.name, observer=sender.name, observed=sender.name
        )
    )
    await db.flush()
    sources = [
        models.Document(
            workspace_name=workspace.name,
            observer=sender.name,
            observed=sender.name,
            session_name=session.name,
            content=content,
            level="explicit",
            embedding=[0.5] * 1536,
        )
        for content in ("alice drinks tea", "alice works nights")
    ]
    db.add_all(sources)
    await db.flush()
    seeded = Seeded(
        workspace=workspace.name,
        session=session.name,
        sender=sender.name,
        scope_peer=scope.name,
        source_ids=[source.id for source in sources],
    )
    await db.commit()
    return seeded


async def _backfill(
    seeded: Seeded,
    monkeypatch: pytest.MonkeyPatch,
    between_plan_and_write: BetweenPlanAndWrite | None = None,
) -> None:
    """Run the backfill task; act once after its plan, before its write."""
    hydrate = scope_backfill._hydrate_chunk_embeddings  # pyright: ignore[reportPrivateUsage]
    pending = [between_plan_and_write] if between_plan_and_write is not None else []

    async def act_then_hydrate(*args: Any, **kwargs: Any) -> None:
        if pending:
            await pending.pop()()
        await hydrate(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(scope_backfill, "_hydrate_chunk_embeddings", act_then_hydrate)
        await process_scope_backfill(
            ScopeBackfillPayload(
                scope_peer=seeded.scope_peer, session_name=seeded.session
            ),
            seeded.workspace,
        )


async def _copies(db: AsyncSession, seeded: Seeded) -> dict[str, bool]:
    """``copied_from`` of each copy in the scope -> whether the copy is live."""
    rows = await db.execute(
        select(models.Document.internal_metadata, models.Document.deleted_at).where(
            models.Document.workspace_name == seeded.workspace,
            models.Document.observer == seeded.scope_peer,
        )
    )
    return {
        row.internal_metadata[COPIED_FROM_KEY]: row.deleted_at is None for row in rows
    }


async def _scope_collections(db: AsyncSession, seeded: Seeded) -> int:
    return int(
        await db.scalar(
            select(func.count())
            .select_from(models.Collection)
            .where(
                models.Collection.workspace_name == seeded.workspace,
                models.Collection.observer == seeded.scope_peer,
            )
        )
        or 0
    )


async def test_an_undisturbed_backfill_copies_every_source(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = await _seed(db_session)
    await _backfill(seeded, monkeypatch)

    assert await _copies(db_session, seeded) == dict.fromkeys(seeded.source_ids, True)
    assert await _scope_collections(db_session, seeded) == 1


async def test_a_session_deleted_after_the_plan_gets_no_copies(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = await _seed(db_session)

    async def delete_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, seeded.workspace, seeded.session)

    with pytest.raises(WriteParentGoneError) as raised:
        await _backfill(seeded, monkeypatch, delete_the_session)

    assert raised.value.scope == "session"
    assert await _copies(db_session, seeded) == {}
    # Neither the scope's collection nor the session was created by the write.
    assert await _scope_collections(db_session, seeded) == 0
    assert (
        await db_session.scalar(
            select(models.Session.id).where(models.Session.name == seeded.session)
        )
        is None
    )


async def test_a_scope_deleted_after_the_plan_gets_no_copies(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = await _seed(db_session)

    async def delete_the_scope_peer() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                update(models.Peer)
                .where(
                    models.Peer.workspace_name == seeded.workspace,
                    models.Peer.name == seeded.scope_peer,
                )
                .values(deleted_at=datetime.now(UTC))
            )
            await db.commit()

    with pytest.raises(WriteParentGoneError) as raised:
        await _backfill(seeded, monkeypatch, delete_the_scope_peer)

    assert raised.value.scope == "peer"
    assert await _copies(db_session, seeded) == {}
    assert await _scope_collections(db_session, seeded) == 0


async def test_a_source_erased_after_the_plan_is_not_copied(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = await _seed(db_session)
    erased, kept = seeded.source_ids

    async def erase_a_source() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_document_by_id(db, seeded.workspace, erased)

    # The whole write is refused: the plan it came from is stale.
    with pytest.raises(WriteGenerationStaleError) as raised:
        await _backfill(seeded, monkeypatch, erase_a_source)
    assert raised.value.scope == "collection"
    assert await _copies(db_session, seeded) == {}

    # The queue runs it again; the new plan no longer holds the erased source.
    await _backfill(seeded, monkeypatch)
    assert await _copies(db_session, seeded) == {kept: True}


async def test_an_erased_source_does_not_restore_its_old_copy(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scope restoration: a re-add clears ``deleted_at`` on the copies a removal left."""
    seeded = await _seed(db_session)
    erased, kept = seeded.source_ids
    await _backfill(seeded, monkeypatch)

    # The session leaves the scope and comes back.
    membership = (
        models.SessionPeer.workspace_name == seeded.workspace,
        models.SessionPeer.session_name == seeded.session,
        models.SessionPeer.peer_name == seeded.scope_peer,
    )
    await db_session.execute(
        update(models.SessionPeer).where(*membership).values(left_at=func.now())
    )
    await db_session.commit()
    await process_scope_removal(
        ScopeRemovalPayload(scope_peer=seeded.scope_peer, session_name=seeded.session),
        seeded.workspace,
    )
    assert await _copies(db_session, seeded) == dict.fromkeys(seeded.source_ids, False)
    await db_session.execute(
        update(models.SessionPeer).where(*membership).values(left_at=None)
    )
    await db_session.commit()

    async def erase_a_source() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_document_by_id(db, seeded.workspace, erased)

    with pytest.raises(WriteGenerationStaleError):
        await _backfill(seeded, monkeypatch, erase_a_source)
    assert await _copies(db_session, seeded) == dict.fromkeys(seeded.source_ids, False)

    # Recomputed: only the copy of the surviving source comes back.
    await _backfill(seeded, monkeypatch)
    assert await _copies(db_session, seeded) == {erased: False, kept: True}


async def test_a_source_consolidated_after_the_plan_is_left_out(
    db_session: AsyncSession, db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dream removed a source. Nothing was erased, so the rest is still copied."""
    seeded = await _seed(db_session)
    consolidated, kept = seeded.source_ids

    async def consolidate_a_source() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_documents(
                db,
                seeded.workspace,
                [consolidated],
                observer=seeded.sender,
                observed=seeded.sender,
            )

    await _backfill(seeded, monkeypatch, consolidate_a_source)

    assert await _copies(db_session, seeded) == {kept: True}


async def test_the_write_commits_once_and_only_after_its_inserts(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write locks the scope membership, then must not commit before it inserts.

    It used to create a missing scope collection through a helper that commits,
    which released the membership lock ahead of the inserts.
    """
    seeded = await _seed(db_session)
    pending_at_commit: list[int] = []
    tracked_db = scope_backfill.tracked_db  # pyright: ignore[reportPrivateLocalImportUsage]

    @asynccontextmanager
    async def recording_tracked_db(
        operation_name: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[AsyncSession]:
        async with tracked_db(operation_name, **kwargs) as db:
            if operation_name == "scope_backfill.write":
                commit = db.commit
                copies = (
                    select(func.count())
                    .select_from(models.Document)
                    .where(
                        models.Document.workspace_name == seeded.workspace,
                        models.Document.observer == seeded.scope_peer,
                    )
                )

                async def counting_commit() -> None:
                    pending_at_commit.append(int(await db.scalar(copies) or 0))
                    await commit()

                db.commit = counting_commit  # type: ignore[method-assign]
            yield db

    async def must_not_be_called(*_: Any, **_kwargs: Any) -> None:
        raise AssertionError("get_or_create_collection commits")

    monkeypatch.setattr(scope_backfill, "tracked_db", recording_tracked_db)
    monkeypatch.setattr(crud, "get_or_create_collection", must_not_be_called)

    await _backfill(seeded, monkeypatch)

    # One commit, and the copies were already written in its transaction.
    assert pending_at_commit == [len(seeded.source_ids)]
    assert await _scope_collections(db_session, seeded) == 1
