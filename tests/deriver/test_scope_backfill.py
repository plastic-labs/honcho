"""Tests for scope backfill-by-copy and removal reconciliation (DEV-1999).

A scope is an observer peer (``scope.<name>``). Adding a session with
pre-existing messages to a scope enqueues a ``scope_backfill`` task; the
handler (``src.deriver.scope_backfill``) copies each observed peer's
explicit-level documents from their global ``(P, P)`` collection into the
scope's ``(scope_peer, P)`` collection, stamping ``copied_from`` for
idempotency, then enqueues a manual omni dream. Removal enqueues
``scope_removal``, which soft-deletes the copies (cascading to dependent
derived documents) and enqueues a card_refresh (rebuild) + omni dream.

These tests exercise the handlers directly (``process_scope_backfill`` /
``process_scope_removal``) against real Collection/Document/Peer rows,
mirroring the fixture style in tests/crud/test_document.py and
tests/dreamer/test_card_refresh.py: rows are created directly via
``db_session`` (never through the cache-backed ``crud.get_or_create_collection``,
which the ``mock_crud_collection_operations`` autouse fixture stubs out to an
unpersisted object for every other test). Fixture data must be *committed*
(not merely flushed) because the handlers run their DB work through
``tracked_db``, which in tests opens a separate session bound to the same
engine (see ``mock_tracked_db_context`` in conftest.py) — a different
connection that cannot see another session's uncommitted writes.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models
from src.deriver import scope_backfill as scope_backfill_mod
from src.deriver.scope_backfill import (
    COPIED_FROM_KEY,
    process_scope_backfill,
    process_scope_removal,
)
from src.schemas import DreamType
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import is_scope_peer, scope_peer_name

_EMBEDDING_DIM = 1536


def _embedding(seed: float = 0.5) -> list[float]:
    return [seed] * _EMBEDDING_DIM


async def _create_peer(db_session: AsyncSession, workspace_name: str) -> models.Peer:
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add(peer)
    await db_session.commit()
    return peer


async def _create_scope_peer(
    db_session: AsyncSession, workspace_name: str, scope_name: str
) -> models.Peer:
    peer = models.Peer(
        name=scope_peer_name(scope_name),
        workspace_name=workspace_name,
        # The authoritative kind flag lives in internal_metadata, which is not
        # user-writable; `configuration` carries only the observe_me knob.
        internal_metadata={"kind": "scope"},
        configuration={"observe_me": False},
    )
    db_session.add(peer)
    await db_session.commit()
    return peer


async def _create_session(
    db_session: AsyncSession, workspace_name: str
) -> models.Session:
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace_name)
    db_session.add(session)
    await db_session.commit()
    return session


async def _join_scope(
    db_session: AsyncSession, workspace_name: str, session_name: str, scope_peer: str
) -> None:
    """Record the scope peer's membership, as the scopes routes do.

    The backfill handler refuses to copy into a scope the session has left, so
    tests driving the handler directly must stand the membership row up.
    """
    db_session.add(
        models.SessionPeer(
            workspace_name=workspace_name,
            session_name=session_name,
            peer_name=scope_peer,
        )
    )
    await db_session.commit()


async def _create_collection(
    db_session: AsyncSession, workspace_name: str, observer: str, observed: str
) -> models.Collection:
    collection = models.Collection(
        workspace_name=workspace_name, observer=observer, observed=observed
    )
    db_session.add(collection)
    await db_session.commit()
    return collection


async def _create_document(
    db_session: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    session_name: str | None,
    content: str = "some observation",
    level: str = "explicit",
    embedding: list[float] | None = None,
    internal_metadata: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
) -> models.Document:
    doc = models.Document(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        content=content,
        level=level,
        session_name=session_name,
        embedding=embedding if embedding is not None else _embedding(),
        internal_metadata=internal_metadata or {},
        source_ids=source_ids,
    )
    db_session.add(doc)
    await db_session.commit()
    return doc


async def _get_docs(
    db_session: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str | None = None,
    include_deleted: bool = True,
) -> list[models.Document]:
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.observer == observer,
    )
    if observed is not None:
        stmt = stmt.where(models.Document.observed == observed)
    if not include_deleted:
        stmt = stmt.where(models.Document.deleted_at.is_(None))
    result = await db_session.execute(stmt)
    return list(result.scalars().all())


async def _dream_items(
    db_session: AsyncSession, workspace_name: str
) -> list[models.QueueItem]:
    result = await db_session.execute(
        select(models.QueueItem).where(
            models.QueueItem.workspace_name == workspace_name,
            models.QueueItem.task_type == "dream",
        )
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# 1. Backfill copies exactly the target session's explicit docs
# ---------------------------------------------------------------------------


async def test_backfill_copies_only_target_session_explicit_docs(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)

    target_session = await _create_session(db_session, workspace_name)
    other_session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, target_session.name, scope_peer.name)

    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    # Destination collection: crud.get_or_create_collection is stubbed to an
    # unpersisted object by the autouse mock_crud_collection_operations
    # fixture, so the scope's own collection must already exist for the
    # copied Document rows' FK to resolve.
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )

    # In-scope: the target session's explicit doc.
    target_doc = await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=target_session.name,
        content="target session explicit fact",
        embedding=_embedding(0.7),
    )
    # Out-of-scope: another session's explicit doc.
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=other_session.name,
        content="other session explicit fact",
    )
    # Out-of-scope: a derived (non-explicit) doc for the target session.
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=target_session.name,
        content="deductive fact",
        level="deductive",
    )

    await process_scope_backfill(
        ScopeBackfillPayload(
            scope_peer=scope_peer.name, session_name=target_session.name
        ),
        workspace_name,
    )

    copies = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    assert len(copies) == 1
    copy = copies[0]
    assert copy.content == "target session explicit fact"
    assert copy.level == "explicit"
    assert copy.session_name == target_session.name
    assert copy.internal_metadata[COPIED_FROM_KEY] == target_doc.id
    assert list(copy.embedding) == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        _embedding(0.7)
    )
    assert copy.deleted_at is None


# ---------------------------------------------------------------------------
# 2. Idempotency
# ---------------------------------------------------------------------------


async def test_backfill_processed_twice_is_idempotent(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
    )

    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )

    payload = ScopeBackfillPayload(
        scope_peer=scope_peer.name, session_name=session.name
    )
    await process_scope_backfill(payload, workspace_name)
    await process_scope_backfill(payload, workspace_name)

    copies = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    assert len(copies) == 1


async def test_add_remove_readd_converges_on_one_live_copy(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
    )

    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )

    backfill_payload = ScopeBackfillPayload(
        scope_peer=scope_peer.name, session_name=session.name
    )
    removal_payload = ScopeRemovalPayload(
        scope_peer=scope_peer.name, session_name=session.name
    )

    # add
    await process_scope_backfill(backfill_payload, workspace_name)
    # remove
    await process_scope_removal(removal_payload, workspace_name)
    live = await _get_docs(
        db_session,
        workspace_name,
        observer=scope_peer.name,
        observed=sender.name,
        include_deleted=False,
    )
    assert live == []
    # re-add
    await process_scope_backfill(backfill_payload, workspace_name)

    all_copies = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    live_copies = [d for d in all_copies if d.deleted_at is None]
    assert len(all_copies) == 1  # restored, not duplicated
    assert len(live_copies) == 1


async def test_backfill_skips_a_session_that_left_the_scope(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """A removal that lands first must not be undone by a queued backfill.

    scope_backfill and scope_removal carry different work-unit keys, so nothing
    orders them: add-then-remove can leave a backfill queued after removal has
    already swept the scope.
    """
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
    )

    # The session leaves the scope before the queued backfill is drained.
    await db_session.execute(
        update(models.SessionPeer)
        .where(
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.session_name == session.name,
            models.SessionPeer.peer_name == scope_peer.name,
        )
        .values(left_at=func.now())
    )
    await db_session.commit()

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    assert (
        await _get_docs(
            db_session, workspace_name, observer=scope_peer.name, observed=sender.name
        )
        == []
    )
    # No status entry either: removal cleared it, and a skipped backfill must
    # not resurrect the session in the scope's status map.
    # Names held as plain strings: expire_all() below would make reading them
    # off the ORM instances trigger a lazy reload mid-assertion.
    scope_peer_name_str, session_name = scope_peer.name, session.name
    db_session.expire_all()
    peer = await db_session.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == scope_peer_name_str)
    )
    assert peer is not None
    assert session_name not in peer.internal_metadata.get("backfill_status", {})


@pytest.mark.asyncio
async def test_copy_chunk_membership_lock_blocks_leave_until_write_commits(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
):
    """A concurrent leave cannot commit between membership check and inserts."""
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    source = await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
        content="locked membership fact",
    )

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    leave_finished = asyncio.Event()
    leave_task_box: dict[str, asyncio.Task[None]] = {}

    async def concurrent_leave() -> None:
        async with factory() as leave_db:
            await leave_db.execute(
                update(models.SessionPeer)
                .where(
                    models.SessionPeer.workspace_name == workspace_name,
                    models.SessionPeer.session_name == session.name,
                    models.SessionPeer.peer_name == scope_peer.name,
                    models.SessionPeer.left_at.is_(None),
                )
                .values(left_at=func.now())
            )
            await leave_db.commit()
        leave_finished.set()

    original_tracked_db = scope_backfill_mod.tracked_db  # pyright: ignore[reportPrivateLocalImportUsage]

    @asynccontextmanager
    async def tracked_db_with_leave_race(
        operation_name: str | None = None, *, read_only: bool = False
    ) -> AsyncGenerator[AsyncSession]:
        async with original_tracked_db(operation_name, read_only=read_only) as db:
            if operation_name == "scope_backfill.write":
                real_scalar = db.scalar
                raced = False

                async def scalar_then_race(statement: Any, *args: Any, **kwargs: Any):
                    nonlocal raced
                    result = await real_scalar(statement, *args, **kwargs)
                    if not raced and result is not None:
                        raced = True
                        leave_task_box["task"] = asyncio.create_task(concurrent_leave())
                        # Leave's UPDATE must block on this txn's row lock.
                        for _ in range(50):
                            await asyncio.sleep(0.01)
                            if leave_task_box["task"].done():
                                break
                        assert not leave_task_box["task"].done()
                    return result

                db.scalar = scalar_then_race  # type: ignore[method-assign]
            yield db

    monkeypatch.setattr(scope_backfill_mod, "tracked_db", tracked_db_with_leave_race)

    ok = await scope_backfill_mod._copy_chunk(  # pyright: ignore[reportPrivateUsage]
        workspace_name,
        scope_peer.name,
        session.name,
        [
            scope_backfill_mod._CopySpec(  # pyright: ignore[reportPrivateUsage]
                observed=sender.name,
                source_id=source.id,
                content=source.content,
                embedding=None,
                internal_metadata={},
                times_derived=1,
                source_ids=None,
                session_name=session.name,
            )
        ],
        store_in_postgres=True,
    )
    assert ok is True

    leave_task = leave_task_box["task"]
    await asyncio.wait_for(leave_task, timeout=2.0)
    assert leave_finished.is_set()

    copies = await _get_docs(
        db_session,
        workspace_name,
        observer=scope_peer.name,
        observed=sender.name,
        include_deleted=False,
    )
    assert len(copies) == 1
    assert copies[0].internal_metadata.get(COPIED_FROM_KEY) == source.id

    membership = await db_session.scalar(
        select(models.SessionPeer.left_at).where(
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.session_name == session.name,
            models.SessionPeer.peer_name == scope_peer.name,
        )
    )
    assert membership is not None


# ---------------------------------------------------------------------------
# 3. Multi-peer session
# ---------------------------------------------------------------------------


async def test_backfill_multi_peer_session_copies_into_right_collections(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, peer_a = sample_data
    workspace_name = test_workspace.name
    peer_b = await _create_peer(db_session, workspace_name)
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)

    for peer in (peer_a, peer_b):
        await _create_collection(
            db_session, workspace_name, observer=peer.name, observed=peer.name
        )
        await _create_collection(
            db_session, workspace_name, observer=scope_peer.name, observed=peer.name
        )
        await _create_document(
            db_session,
            workspace_name,
            observer=peer.name,
            observed=peer.name,
            session_name=session.name,
            content=f"fact about {peer.name}",
        )

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    copies_a = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=peer_a.name
    )
    copies_b = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=peer_b.name
    )
    assert len(copies_a) == 1
    assert copies_a[0].content == f"fact about {peer_a.name}"
    assert len(copies_b) == 1
    assert copies_b[0].content == f"fact about {peer_b.name}"


# ---------------------------------------------------------------------------
# 4. Removal cascade
# ---------------------------------------------------------------------------


async def test_removal_cascades_to_dependent_derived_docs_only(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
    )

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )
    [copy] = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )

    # A derived doc resting on the copy's evidence -> must be cascaded.
    dependent = await _create_document(
        db_session,
        workspace_name,
        observer=scope_peer.name,
        observed=sender.name,
        session_name=None,
        content="deduction resting on removed evidence",
        level="deductive",
        source_ids=[copy.id],
    )
    # An unrelated derived doc in the same collection -> must survive.
    unrelated = await _create_document(
        db_session,
        workspace_name,
        observer=scope_peer.name,
        observed=sender.name,
        session_name=None,
        content="unrelated deduction",
        level="deductive",
        source_ids=["some-other-doc-id-not-removed"],
    )

    copy_id, dependent_id, unrelated_id = copy.id, dependent.id, unrelated.id

    await process_scope_removal(
        ScopeRemovalPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    # process_scope_removal runs on a separate tracked_db session (a
    # different connection). Query raw columns rather than full ORM entities
    # so this session's identity map (holding the pre-removal `copy` /
    # `dependent` / `unrelated` instances) can't hand back stale, expired
    # attributes.
    result = await db_session.execute(
        select(models.Document.id, models.Document.deleted_at).where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == scope_peer.name,
            models.Document.observed == sender.name,
        )
    )
    deleted_at_by_id = {row[0]: row[1] for row in result.all()}
    assert deleted_at_by_id[copy_id] is not None
    assert deleted_at_by_id[dependent_id] is not None
    assert deleted_at_by_id[unrelated_id] is None


# ---------------------------------------------------------------------------
# 5. Dream enqueues
# ---------------------------------------------------------------------------


async def test_backfill_enqueues_manual_omni_dream(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
    )

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    dreams = await _dream_items(db_session, workspace_name)
    assert len(dreams) == 1
    payload = dreams[0].payload
    assert payload["dream_type"] == DreamType.OMNI.value
    assert payload["observer"] == scope_peer.name
    assert payload["observed"] == sender.name
    assert payload["trigger_reason"] == "scope_backfill"
    assert payload.get("rebuild", False) is False


async def test_removal_enqueues_card_refresh_rebuild_and_omni_dream(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Removal's own dream enqueues, isolated from backfill's.

    The scope's copy is created directly (as if an earlier backfill already
    ran and its dream was drained by the deriver) rather than by calling
    process_scope_backfill first: enqueue_dream dedupes on work_unit_key, so
    a still-pending omni dream from an immediately-preceding backfill would
    silently swallow removal's own omni enqueue and make this test couple to
    that unrelated dedup behavior instead of testing removal in isolation.
    """
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=scope_peer.name,
        observed=sender.name,
        session_name=session.name,
        internal_metadata={COPIED_FROM_KEY: "some-source-doc-id"},
    )

    await process_scope_removal(
        ScopeRemovalPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    dreams = await _dream_items(db_session, workspace_name)
    removal_dreams = [
        d for d in dreams if d.payload.get("trigger_reason") == "scope_removal"
    ]
    assert len(removal_dreams) == 2

    by_type = {d.payload["dream_type"]: d.payload for d in removal_dreams}
    assert DreamType.CARD_REFRESH.value in by_type
    assert DreamType.OMNI.value in by_type
    card_refresh_payload = by_type[DreamType.CARD_REFRESH.value]
    assert card_refresh_payload["rebuild"] is True
    assert card_refresh_payload["observer"] == scope_peer.name
    assert card_refresh_payload["observed"] == sender.name


# ---------------------------------------------------------------------------
# 6. Status endpoint
# ---------------------------------------------------------------------------


async def test_status_reflects_pending_then_completed(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())

    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes", json={"id": scope_name}
    )
    assert response.status_code == 201
    scope_peer_full_name = scope_peer_name(scope_name)

    session_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/sessions",
        json={"id": session_name, "peers": {sender.name: {}}},
    )
    assert response.status_code == 201

    message = models.Message(
        workspace_name=workspace_name,
        session_name=session_name,
        peer_name=sender.name,
        content="hello from before the scope existed",
        public_id=generate_nanoid(),
        seq_in_session=1,
        token_count=5,
    )
    db_session.add(message)
    await db_session.commit()

    # The message's explicit document (normally produced by the deriver) —
    # created directly since the deriver isn't run in this test.
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session_name,
    )

    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/sessions",
        json={"session_ids": [session_name]},
    )
    assert response.status_code == 204, response.text

    # Destination collection: crud.get_or_create_collection is stubbed to an
    # unpersisted object by the autouse fixture, so it must pre-exist.
    await _create_collection(
        db_session, workspace_name, observer=scope_peer_full_name, observed=sender.name
    )

    status_url = f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/status"
    response = client.get(status_url)
    assert response.status_code == 200, response.text
    backfill_status = response.json()["backfill_status"]
    assert backfill_status[session_name]["state"] == "pending"

    # Simulate the deriver picking up the enqueued task.
    await process_scope_backfill(
        ScopeBackfillPayload(
            scope_peer=scope_peer_full_name, session_name=session_name
        ),
        workspace_name,
    )

    response = client.get(status_url)
    assert response.status_code == 200, response.text
    backfill_status = response.json()["backfill_status"]
    assert backfill_status[session_name]["state"] == "completed"
    assert backfill_status[session_name]["docs_copied"] == 1


async def test_backfill_re_embeds_sources_with_null_embeddings(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Source rows carry no embedding on external-store deployments.

    Phase 2 re-embeds those (embedding API only) and pairs results back with
    strict=True, so a mis-pairing would raise rather than silently mismatch.
    """
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    source = await _create_document(
        db_session,
        workspace_name,
        observer=sender.name,
        observed=sender.name,
        session_name=session.name,
        content="fact whose vector lives in the external store",
    )
    source.embedding = None
    await db_session.commit()

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    [copy] = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    assert copy.embedding is not None
    assert len(copy.embedding) == _EMBEDDING_DIM


async def test_backfill_failure_records_failed_status(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    test_workspace, _ = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)

    # Plain strings: the ORM instances are expired below (see the same guard in
    # test_backfill_status_writes_preserve_the_scope_kind_flag).
    scope_peer_name_str, session_name = scope_peer.name, session.name

    async def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("copy phase blew up")

    monkeypatch.setattr("src.deriver.scope_backfill._run_backfill", boom)

    with pytest.raises(RuntimeError):
        await process_scope_backfill(
            ScopeBackfillPayload(
                scope_peer=scope_peer_name_str, session_name=session_name
            ),
            workspace_name,
        )

    db_session.expire_all()
    peer = await db_session.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == scope_peer_name_str)
    )
    assert peer is not None
    assert peer.internal_metadata["backfill_status"][session_name]["state"] == "failed"


# ---------------------------------------------------------------------------
# 7. Route wiring: add-sessions enqueues backfill only when messages exist
# ---------------------------------------------------------------------------


async def test_add_sessions_enqueues_backfill_only_when_session_has_messages(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    assert (
        client.post(
            f"/v3/workspaces/{workspace_name}/scopes", json={"id": scope_name}
        ).status_code
        == 201
    )

    # Session with a pre-existing message.
    session_with_messages = str(generate_nanoid())
    assert (
        client.post(
            f"/v3/workspaces/{workspace_name}/sessions",
            json={"id": session_with_messages, "peers": {sender.name: {}}},
        ).status_code
        == 201
    )
    message = models.Message(
        workspace_name=workspace_name,
        session_name=session_with_messages,
        peer_name=sender.name,
        content="already said something",
        public_id=generate_nanoid(),
        seq_in_session=1,
        token_count=5,
    )
    db_session.add(message)
    await db_session.commit()

    # Empty session, no messages.
    empty_session = str(generate_nanoid())
    assert (
        client.post(
            f"/v3/workspaces/{workspace_name}/sessions",
            json={"id": empty_session},
        ).status_code
        == 201
    )

    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/sessions",
        json={"session_ids": [session_with_messages, empty_session]},
    )
    assert response.status_code == 204, response.text

    result = await db_session.execute(
        select(models.QueueItem).where(
            models.QueueItem.workspace_name == workspace_name,
            models.QueueItem.task_type == "scope_backfill",
        )
    )
    backfill_items = list(result.scalars().all())
    assert len(backfill_items) == 1
    assert backfill_items[0].payload["session_name"] == session_with_messages


# ---------------------------------------------------------------------------
# The scope `kind` flag and the backfill status map share the scope peer's
# internal_metadata. Every write to that column must be a JSONB merge scoped to
# the backfill key; a wholesale assignment would drop the flag and silently turn
# the peer back into an ordinary one — invisible until some later read stopped
# recognising it as a scope.
# ---------------------------------------------------------------------------


async def test_backfill_status_writes_preserve_the_scope_kind_flag(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Status writes must not clobber the authoritative kind flag.

    Both live in internal_metadata, so this pins the one property that makes
    them able to coexist. Covers the whole lifecycle, because a wholesale write
    could be introduced at any single step: pending, completed, then cleared.
    """
    test_workspace, _ = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    session_name = str(generate_nanoid())
    await _create_scope_peer(db_session, workspace_name, scope_name)
    # Held as a plain string: the ORM instance is expired below on every check,
    # so reading an attribute off it would trigger a reload mid-assertion.
    backing_peer = scope_peer_name(scope_name)

    async def assert_still_a_scope(stage: str) -> dict[str, Any]:
        db_session.expire_all()
        refreshed = await db_session.scalar(
            select(models.Peer)
            .where(models.Peer.workspace_name == workspace_name)
            .where(models.Peer.name == backing_peer)
        )
        assert refreshed is not None
        assert is_scope_peer(refreshed.name, refreshed.internal_metadata), (
            f"the peer stopped being a scope after {stage}: "
            f"internal_metadata={refreshed.internal_metadata!r}"
        )
        # And the facade still resolves it, which is what actually breaks:
        # get_scope_or_raise 404s on a peer that has lost the flag.
        resolved = await crud.get_scope_or_raise(db_session, workspace_name, scope_name)
        assert resolved.name == backing_peer
        return refreshed.internal_metadata

    await assert_still_a_scope("creation")

    await crud.update_scope_backfill_status(
        db_session,
        workspace_name,
        backing_peer,
        session_name,
        state="pending",
    )
    await db_session.commit()
    metadata = await assert_still_a_scope("a pending status write")
    assert metadata["backfill_status"][session_name]["state"] == "pending"

    await crud.update_scope_backfill_status(
        db_session,
        workspace_name,
        backing_peer,
        session_name,
        state="completed",
        docs_copied=3,
    )
    await db_session.commit()
    metadata = await assert_still_a_scope("a completed status write")
    assert metadata["backfill_status"][session_name]["docs_copied"] == 3

    await crud.clear_scope_backfill_status(
        db_session, workspace_name, backing_peer, session_name
    )
    await db_session.commit()
    metadata = await assert_still_a_scope("clearing the status")
    assert session_name not in metadata.get("backfill_status", {})


@pytest.mark.asyncio
async def test_backfill_embeds_and_writes_in_bounded_chunks(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    """Phases 2-4 run per chunk, so a large session never holds every vector."""
    from src.deriver import scope_backfill
    from src.embedding_client import embedding_client

    test_workspace, sender = sample_data
    workspace_name = test_workspace.name
    scope_name = str(generate_nanoid())
    scope_peer = await _create_scope_peer(db_session, workspace_name, scope_name)
    session = await _create_session(db_session, workspace_name)
    await _join_scope(db_session, workspace_name, session.name, scope_peer.name)
    await _create_collection(
        db_session, workspace_name, observer=sender.name, observed=sender.name
    )
    await _create_collection(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    for i in range(3):
        source = await _create_document(
            db_session,
            workspace_name,
            observer=sender.name,
            observed=sender.name,
            session_name=session.name,
            content=f"fact {i}",
        )
        source.embedding = None
    await db_session.commit()

    batch_sizes: list[int] = []
    seen_specs: list[scope_backfill._CopySpec] = []  # pyright: ignore[reportPrivateUsage]
    peak_live_embeddings = 0
    original_embed = embedding_client.simple_batch_embed
    original_copy_chunk = scope_backfill._copy_chunk  # pyright: ignore[reportPrivateUsage]

    async def recording_embed(texts: list[str], **kwargs: Any) -> list[list[float]]:
        batch_sizes.append(len(texts))
        return await original_embed(texts, **kwargs)

    async def counting_copy_chunk(
        ws_name: str,
        peer_name: str,
        sess_name: str,
        plans: list[scope_backfill._CopySpec],  # pyright: ignore[reportPrivateUsage]
        store_in_postgres: bool,
    ) -> bool:
        nonlocal peak_live_embeddings
        seen_specs.extend(plans)
        result = await original_copy_chunk(
            ws_name, peer_name, sess_name, plans, store_in_postgres
        )
        # Sampled after this chunk syncs but before _run_backfill drops its
        # vectors, so every *earlier* chunk must already be cleared and the
        # live count can never exceed one chunk. That drop is the whole
        # memory bound; without it this peaks at 3 instead of 2.
        peak_live_embeddings = max(
            peak_live_embeddings,
            sum(1 for spec in seen_specs if spec.embedding is not None),
        )
        return result

    monkeypatch.setattr(scope_backfill, "BACKFILL_CHUNK_SIZE", 2)
    monkeypatch.setattr(embedding_client, "simple_batch_embed", recording_embed)
    monkeypatch.setattr(scope_backfill, "_copy_chunk", counting_copy_chunk)

    await process_scope_backfill(
        ScopeBackfillPayload(scope_peer=scope_peer.name, session_name=session.name),
        workspace_name,
    )

    assert batch_sizes == [2, 1]
    assert peak_live_embeddings == 2
    copies = await _get_docs(
        db_session, workspace_name, observer=scope_peer.name, observed=sender.name
    )
    assert len(copies) == 3
    assert all(copy.embedding is not None for copy in copies)
