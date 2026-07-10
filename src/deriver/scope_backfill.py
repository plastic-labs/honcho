"""Scope membership reconciliation jobs (DEV-1999, part of the Scopes RFC).

Two queue task handlers, dispatched from ``consumer.process_item``:

- ``scope_backfill`` — a session was added to a scope that already had
  messages. Copy the session's *explicit* documents from each sender peer's
  global ``(P, P)`` collection into the scope's ``(scope_peer, P)`` collection,
  then enqueue a manual omni dream per touched collection to rebuild the
  scope's higher-order layer and card.
- ``scope_removal`` — a session was removed from a scope. Soft-delete the
  session's explicit documents from the scope's collections, cascade the
  soft-delete to derived documents whose support left the scope (fail-closed),
  then enqueue a ``card_refresh`` dream with ``rebuild=True`` plus a manual
  omni dream per touched collection.

Zero LLM re-derivation: explicit-level documents are session-pure and
identical across observer collections (the DEV-2000 invariant), so retroactive
membership is pure row copying. The only external call is an embedding lookup
for source rows whose embedding column is NULL — embedding API only, never an
LLM.

DB sessions are never held across embedding or vector-store calls: the
handlers run in phases (plan → embed → write → sync), each phase opening its
own short-lived session.
"""

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.sql.functions import func

from src import crud, models
from src.config import settings
from src.crud.scope import ScopeBackfillState
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.schemas import DreamType
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import is_scope_peer_name
from src.vector_store import VectorRecord, get_external_vector_store

logger = logging.getLogger(__name__)

# internal_metadata key linking a backfilled copy to the global document it
# was copied from. The presence of this key is the idempotency marker.
COPIED_FROM_KEY = "copied_from"


def _store_embeddings_in_postgres() -> bool:
    """Whether document embeddings are persisted to the postgres column.

    True when TYPE=pgvector OR still migrating (dual-write) — mirrors
    ``crud.document.create_documents``.
    """
    return (
        settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED
    )


def _embedding_as_list(embedding: Any) -> list[float] | None:
    """Normalize a pgvector column value (numpy array or None) to a list."""
    if embedding is None:
        return None
    return list(embedding)


@dataclass
class _CopySpec:
    """A planned copy of one global explicit document into a scope collection.

    ``restore_document_id`` points at an existing soft-deleted copy to restore
    when set; for new copies it is filled with the inserted row's id after the
    write phase.
    """

    observed: str
    source_id: str
    content: str
    embedding: list[float] | None
    internal_metadata: dict[str, Any]
    times_derived: int
    source_ids: list[str] | None
    session_name: str
    restore_document_id: str | None = None


# ---------------------------------------------------------------------------
# Backfill (session added to a scope)
# ---------------------------------------------------------------------------


async def process_scope_backfill(
    payload: ScopeBackfillPayload, workspace_name: str
) -> None:
    """Process a ``scope_backfill`` queue task.

    Idempotent: a live copy (matched by ``internal_metadata.copied_from``) is
    never duplicated, and a soft-deleted copy left by an earlier removal is
    restored rather than re-inserted, so add→remove→re-add converges on
    exactly one live copy per source document.
    """
    scope_peer = payload.scope_peer
    session_name = payload.session_name

    try:
        docs_copied, touched_observed = await _run_backfill(
            workspace_name, scope_peer, session_name
        )
    except Exception:
        await _write_backfill_status(
            workspace_name, scope_peer, session_name, state="failed"
        )
        raise

    # One manual (gate-bypassing) omni dream per touched collection builds the
    # scope's higher-order layer and card. enqueue_dream dedupes on the
    # work-unit key, so a batch of backfills collapses to one dream each.
    from src.deriver.enqueue import enqueue_dream

    for observed in sorted(touched_observed):
        await enqueue_dream(
            workspace_name,
            observer=scope_peer,
            observed=observed,
            dream_type=DreamType.OMNI,
            trigger_reason="scope_backfill",
        )

    await _write_backfill_status(
        workspace_name,
        scope_peer,
        session_name,
        state="completed",
        docs_copied=docs_copied,
    )

    logger.info(
        "Scope backfill complete for %s/%s/%s: %d documents copied across %d collections",
        workspace_name,
        scope_peer,
        session_name,
        docs_copied,
        len(touched_observed),
    )


async def _run_backfill(
    workspace_name: str, scope_peer: str, session_name: str
) -> tuple[int, set[str]]:
    """Run the copy itself. Returns (docs copied or restored, touched observed peers)."""
    # Phase 1 (DB): plan. Read the session's explicit documents from each
    # sender's global (P, P) collection, and any existing copies (live or
    # soft-deleted) already in the scope's collections.
    plans: list[_CopySpec] = []
    async with tracked_db("scope_backfill.plan") as db:
        source_result = await db.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace_name,
                models.Document.session_name == session_name,
                models.Document.level == "explicit",
                models.Document.observer == models.Document.observed,
                models.Document.deleted_at.is_(None),
            )
        )
        source_docs = [
            doc
            for doc in source_result.scalars().all()
            # Scope peers never speak and are never observed, but stay
            # defensive: never treat another scope's rows as a source.
            if not is_scope_peer_name(doc.observer)
        ]

        if not source_docs:
            return 0, set()

        # Existing copies in the scope's collections for this session, keyed
        # by (observed, copied_from). Includes soft-deleted rows: those are
        # restore candidates, not blockers.
        copies_result = await db.execute(
            select(models.Document).where(
                models.Document.workspace_name == workspace_name,
                models.Document.observer == scope_peer,
                models.Document.session_name == session_name,
                models.Document.internal_metadata.has_key(COPIED_FROM_KEY),
            )
        )
        live_copies: set[tuple[str, str]] = set()
        soft_deleted_copies: dict[tuple[str, str], str] = {}
        for copy_doc in copies_result.scalars().all():
            key = (copy_doc.observed, str(copy_doc.internal_metadata[COPIED_FROM_KEY]))
            if copy_doc.deleted_at is None:
                live_copies.add(key)
            else:
                soft_deleted_copies.setdefault(key, copy_doc.id)

        for source in source_docs:
            key = (source.observed, source.id)
            if key in live_copies:
                continue
            plans.append(
                _CopySpec(
                    observed=source.observed,
                    source_id=source.id,
                    content=source.content,
                    embedding=_embedding_as_list(source.embedding),
                    internal_metadata=dict(source.internal_metadata),
                    times_derived=source.times_derived,
                    source_ids=list(source.source_ids)
                    if source.source_ids is not None
                    else None,
                    session_name=session_name,
                    restore_document_id=soft_deleted_copies.get(key),
                )
            )

    if not plans:
        return 0, set()

    # Phase 2 (no DB): fill missing embeddings. Source rows have NULL
    # embeddings on external-store deployments (and soft-deleted copies may
    # have lost their vectors) — re-embed via the embedding API only; no LLM.
    missing = [spec for spec in plans if spec.embedding is None]
    if missing:
        logger.info(
            "Scope backfill re-embedding %d documents with NULL embeddings for %s/%s/%s (embedding API only, no LLM)",
            len(missing),
            workspace_name,
            scope_peer,
            session_name,
        )
        embeddings = await embedding_client.simple_batch_embed(
            [spec.content for spec in missing]
        )
        for spec, embedding in zip(missing, embeddings, strict=True):
            spec.embedding = embedding

    # Phase 3 (DB): write the copies.
    store_in_postgres = _store_embeddings_in_postgres()
    touched_observed = {spec.observed for spec in plans}
    new_rows: list[models.Document] = []
    async with tracked_db("scope_backfill.write") as db:
        for observed in sorted(touched_observed):
            await crud.get_or_create_collection(
                db, workspace_name, observer=scope_peer, observed=observed
            )

        restore_ids: list[str] = []
        for spec in plans:
            if spec.restore_document_id is not None:
                restore_ids.append(spec.restore_document_id)
                continue
            row = models.Document(
                workspace_name=workspace_name,
                observer=scope_peer,
                observed=spec.observed,
                content=spec.content,
                level="explicit",
                times_derived=spec.times_derived,
                internal_metadata={
                    **spec.internal_metadata,
                    COPIED_FROM_KEY: spec.source_id,
                },
                session_name=spec.session_name,
                source_ids=spec.source_ids,
                embedding=spec.embedding if store_in_postgres else None,
            )
            row.sync_state = "pending"
            new_rows.append(row)

        db.add_all(new_rows)
        if restore_ids:
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(restore_ids))
                .values(deleted_at=None, sync_state="pending")
            )
        await db.commit()

        # IDs are generated client-side and remain accessible post-commit
        # (expire_on_commit=False), mirroring crud.document.create_documents.
        for spec, row in zip(
            [s for s in plans if s.restore_document_id is None], new_rows, strict=True
        ):
            spec.restore_document_id = row.id

    copied_ids = [
        spec.restore_document_id
        for spec in plans
        if spec.restore_document_id is not None
    ]

    # Phase 4: sync to the external vector store (or mark synced in pgvector
    # mode). Failures leave rows in sync_state='pending' for the reconciler.
    await _sync_copies_to_vector_store(workspace_name, scope_peer, plans, copied_ids)

    return len(plans), touched_observed


async def _sync_copies_to_vector_store(
    workspace_name: str,
    scope_peer: str,
    plans: list[_CopySpec],
    copied_ids: list[str],
) -> None:
    """Mirror the document-create sync path for the backfilled copies."""
    external_vector_store = get_external_vector_store()

    if external_vector_store is None:
        # pgvector mode: embeddings live in the postgres column; nothing to sync.
        async with tracked_db("scope_backfill.mark_synced") as db:
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(copied_ids))
                .values(sync_state="synced", last_sync_at=func.now(), sync_attempts=0)
            )
            await db.commit()
        return

    by_observed: dict[str, list[_CopySpec]] = {}
    for spec in plans:
        by_observed.setdefault(spec.observed, []).append(spec)

    synced_ids: list[str] = []
    failed_ids: list[str] = []
    for observed, specs in by_observed.items():
        namespace = external_vector_store.get_vector_namespace(
            "document", workspace_name, scope_peer, observed
        )
        records = [
            VectorRecord(
                id=spec.restore_document_id,
                embedding=spec.embedding,
                metadata={
                    "workspace_name": workspace_name,
                    "observer": scope_peer,
                    "observed": observed,
                    "session_name": spec.session_name,
                    "level": "explicit",
                },
            )
            for spec in specs
            if spec.restore_document_id is not None and spec.embedding is not None
        ]
        ids = [record.id for record in records]
        try:
            await external_vector_store.upsert_many(namespace, records)
            synced_ids.extend(ids)
        except Exception:
            logger.exception(
                "Failed to upsert backfilled vectors to %s; leaving docs pending for the reconciler",
                namespace,
            )
            failed_ids.extend(ids)

    async with tracked_db("scope_backfill.sync_state") as db:
        if synced_ids:
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(synced_ids))
                .values(sync_state="synced", last_sync_at=func.now(), sync_attempts=0)
            )
        if failed_ids:
            await db.execute(
                update(models.Document)
                .where(models.Document.id.in_(failed_ids))
                .values(
                    sync_attempts=models.Document.sync_attempts + 1,
                    last_sync_at=func.now(),
                )
            )
        await db.commit()


# ---------------------------------------------------------------------------
# Removal reconciliation (session removed from a scope)
# ---------------------------------------------------------------------------


async def process_scope_removal(
    payload: ScopeRemovalPayload, workspace_name: str
) -> None:
    """Process a ``scope_removal`` queue task.

    Soft-deletes (fail-closed) rather than hard-deletes: the reconciler's
    soft-delete sweep handles vector cleanup and eventual hard deletion, and a
    later re-add restores the same rows.
    """
    scope_peer = payload.scope_peer
    session_name = payload.session_name

    removed_by_observed: dict[str, list[str]] = {}
    async with tracked_db("scope_removal") as db:
        observed_result = await db.execute(
            select(models.Collection.observed).where(
                models.Collection.workspace_name == workspace_name,
                models.Collection.observer == scope_peer,
            )
        )
        for observed in [row[0] for row in observed_result.all()]:
            explicit_stmt = (
                update(models.Document)
                .where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.observer == scope_peer,
                    models.Document.observed == observed,
                    models.Document.session_name == session_name,
                    models.Document.level == "explicit",
                    models.Document.deleted_at.is_(None),
                )
                .values(deleted_at=func.now())
                .returning(models.Document.id)
            )
            frontier = [row[0] for row in (await db.execute(explicit_stmt)).all()]
            all_removed = list(frontier)

            # Fail-closed cascade: soft-delete derived documents whose support
            # (source_ids) intersects anything removed, transitively — a
            # deduction resting on removed evidence must leave with it, and so
            # must an induction resting on that deduction.
            while frontier:
                derived_stmt = (
                    update(models.Document)
                    .where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == scope_peer,
                        models.Document.observed == observed,
                        models.Document.level != "explicit",
                        models.Document.deleted_at.is_(None),
                        models.Document.source_ids.has_any(array(frontier)),
                    )
                    .values(deleted_at=func.now())
                    .returning(models.Document.id)
                )
                frontier = [row[0] for row in (await db.execute(derived_stmt)).all()]
                all_removed.extend(frontier)

            if all_removed:
                removed_by_observed[observed] = all_removed

        # The session left the scope, so its backfill status entry is moot;
        # clearing it keeps a later re-add starting from a fresh "pending".
        await crud.clear_scope_backfill_status(
            db, workspace_name, scope_peer, session_name
        )
        await db.commit()
    await crud.invalidate_scope_peer_cache(workspace_name, scope_peer)

    # Delete the vectors eagerly so recall can't surface removed memory while
    # waiting for the reconciler sweep (which remains the backstop on failure).
    external_vector_store = get_external_vector_store()
    if external_vector_store is not None:
        for observed, removed_ids in removed_by_observed.items():
            namespace = external_vector_store.get_vector_namespace(
                "document", workspace_name, scope_peer, observed
            )
            try:
                await external_vector_store.delete_many(namespace, removed_ids)
            except Exception:
                logger.exception(
                    "Failed to delete removed vectors from %s; reconciler sweep will retry",
                    namespace,
                )

    # Rebuild what remains: the card must be regenerated from remaining
    # evidence only (rebuild=True drops the stale card from the prompt), and a
    # manual omni dream rebuilds the higher-order structure.
    from src.deriver.enqueue import enqueue_dream

    for observed in sorted(removed_by_observed):
        await enqueue_dream(
            workspace_name,
            observer=scope_peer,
            observed=observed,
            dream_type=DreamType.CARD_REFRESH,
            rebuild=True,
            trigger_reason="scope_removal",
        )
        await enqueue_dream(
            workspace_name,
            observer=scope_peer,
            observed=observed,
            dream_type=DreamType.OMNI,
            trigger_reason="scope_removal",
        )

    logger.info(
        "Scope removal reconciliation complete for %s/%s/%s: %d documents soft-deleted across %d collections",
        workspace_name,
        scope_peer,
        session_name,
        sum(len(ids) for ids in removed_by_observed.values()),
        len(removed_by_observed),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _write_backfill_status(
    workspace_name: str,
    scope_peer: str,
    session_name: str,
    *,
    state: ScopeBackfillState,
    docs_copied: int | None = None,
) -> None:
    """Best-effort status write in its own short-lived session."""
    try:
        async with tracked_db("scope_backfill.status") as db:
            await crud.update_scope_backfill_status(
                db,
                workspace_name,
                scope_peer,
                session_name,
                state=state,
                docs_copied=docs_copied,
            )
            await db.commit()
        await crud.invalidate_scope_peer_cache(workspace_name, scope_peer)
    except Exception:
        # Never mask the underlying failure (or fail a completed backfill)
        # over a status bookkeeping write.
        logger.exception(
            "Failed to write scope backfill status %s for %s/%s/%s",
            state,
            workspace_name,
            scope_peer,
            session_name,
        )
