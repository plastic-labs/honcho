"""Tests for scope backfill-by-copy and removal reconciliation (DEV-1999).

A scope is an observer peer (``scope__<name>``). Adding a session with
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

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.scope_backfill import (
    COPIED_FROM_KEY,
    process_scope_backfill,
    process_scope_removal,
)
from src.schemas import DreamType
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import scope_peer_name

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
        configuration={"kind": "scope", "observe_me": False},
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
    assert response.status_code == 200

    # Destination collection: crud.get_or_create_collection is stubbed to an
    # unpersisted object by the autouse fixture, so it must pre-exist.
    await _create_collection(
        db_session, workspace_name, observer=scope_peer_full_name, observed=sender.name
    )

    status_url = f"/v3/workspaces/{workspace_name}/scopes/{scope_name}/status"
    response = client.get(status_url)
    assert response.status_code == 200
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
    assert response.status_code == 200
    backfill_status = response.json()["backfill_status"]
    assert backfill_status[session_name]["state"] == "completed"
    assert backfill_status[session_name]["docs_copied"] == 1


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
    assert response.status_code == 200

    result = await db_session.execute(
        select(models.QueueItem).where(
            models.QueueItem.workspace_name == workspace_name,
            models.QueueItem.task_type == "scope_backfill",
        )
    )
    backfill_items = list(result.scalars().all())
    assert len(backfill_items) == 1
    assert backfill_items[0].payload["session_name"] == session_with_messages
