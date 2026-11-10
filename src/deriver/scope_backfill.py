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

from nanoid import generate as generate_nanoid
from sqlalchemy import delete, insert, or_, select, update
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import load_only
from sqlalchemy.sql.functions import func

from src import crud, models
from src.config import settings
from src.crud.document import build_source_message_links
from src.crud.lifecycle_barrier import (
    WriteSnapshot,
    bump_collection_generations,
    capture_write_snapshot,
    current_claim,
    ensure_collections,
    final_write,
    stamp_representations,
)
from src.crud.scope import ScopeBackfillState
from src.dependencies import tracked_db
from src.dreamer.working_memory import EPOCH_KEY, NOTES_KEY
from src.embedding_client import embedding_client
from src.exceptions import (
    LifecycleBarrierError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)
from src.schemas import DreamType
from src.utils.queue_payload import ScopeBackfillPayload, ScopeRemovalPayload
from src.utils.scopes import is_scope_peer_name
from src.vector_store import VectorRecord, get_external_vector_store

logger = logging.getLogger(__name__)

# internal_metadata key linking a backfilled copy to the global document it
# was copied from. The presence of this key is the idempotency marker.
COPIED_FROM_KEY = "copied_from"

# Specs embedded, written, and synced per pass. Bounds the live embeddings
# (~40KB each as Python floats) so a large session cannot OOM the deriver.
BACKFILL_CHUNK_SIZE = 500


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
    source_message_ids: list[str] | None
    session_name: str
    restore_document_id: str | None = None
    # Set by the write phase once the copy was inserted or restored. A spec
    # whose source was consolidated away since the plan is left unwritten.
    written: bool = False


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
        result = await _run_backfill(workspace_name, scope_peer, session_name)
    except LifecycleBarrierError:
        # Not a failed backfill: a deletion landed while it ran. The queue
        # drops the task when the session or scope is gone, and runs it again
        # from the current documents when only its plan was stale.
        raise
    except Exception:
        await _write_backfill_status(
            workspace_name, scope_peer, session_name, state="failed"
        )
        raise

    if result is None:
        # The session left the scope before this task was drained: removal
        # already cleared the copies and the status entry, so there is nothing
        # to backfill and nothing to record.
        logger.info(
            "Scope backfill skipped for %s/%s/%s: membership is no longer active",
            workspace_name,
            scope_peer,
            session_name,
        )
        return
    docs_copied, touched_observed = result

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
) -> tuple[int, set[str]] | None:
    """Run the copy itself.

    Returns (docs copied or restored, touched observed peers), or ``None`` if
    the session is no longer a member of the scope.
    """
    # Phase 1 (DB): plan. Read the session's explicit documents from each
    # sender's global (P, P) collection, and any existing copies (live or
    # soft-deleted) already in the scope's collections.
    plans: list[_CopySpec] = []
    async with tracked_db("scope_backfill.plan") as db:
        # Lifecycle barrier: capture before the plan reads any document. The
        # copies depend on the session, the scope peer, and on each sender's
        # own (P, P) collection, whose generation tells a later write that a
        # source it planned to copy was erased.
        barrier = await capture_write_snapshot(
            db,
            workspace_name,
            session_name=session_name,
            peer_names=[scope_peer],
            claim=current_claim(),
        )
        senders = await db.scalars(
            select(models.Document.observer)
            .where(
                models.Document.tenant_id == barrier.tenant_id,
                models.Document.workspace_name == workspace_name,
                models.Document.session_name == session_name,
                models.Document.level == "explicit",
                models.Document.observer == models.Document.observed,
                models.Document.deleted_at.is_(None),
            )
            .distinct()
        )
        for sender in sorted(set(senders)):
            if is_scope_peer_name(sender):
                continue
            try:
                barrier = await stamp_representations(
                    db, barrier, [(sender, sender), (scope_peer, sender)]
                )
            except WriteParentGoneError:
                # The sender was deleted: its conclusions are not copied.
                continue
        stamped_senders = {
            c.observer for c in barrier.collections if c.observer == c.observed
        }

        source_result = await db.execute(
            select(models.Document)
            .options(
                load_only(
                    models.Document.id,
                    models.Document.workspace_name,
                    models.Document.observer,
                    models.Document.observed,
                    models.Document.content,
                    models.Document.level,
                    models.Document.times_derived,
                    models.Document.internal_metadata,
                    models.Document.session_name,
                    # source_ids falls back to this column for undrained rows.
                    models.Document.legacy_source_ids,
                    models.Document.deleted_at,
                )
            )
            .where(
                models.Document.tenant_id == models.scoped_tenant_id(),
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
        if any(doc.observer not in stamped_senders for doc in source_docs):
            # A sender's first conclusions, or a deleted sender's, appeared
            # after the capture: this plan has no generation to check them
            # against. Plan again.
            raise WriteGenerationStaleError(
                "collection",
                f"Session {session_name} gained a sender since the backfill began",
            )

        if not source_docs:
            return 0, set()

        # Existing copies in the scope's collections for this session, keyed
        # by (observed, copied_from). Includes soft-deleted rows: those are
        # restore candidates, not blockers.
        copies_result = await db.execute(
            select(models.Document)
            .options(
                load_only(
                    models.Document.id,
                    models.Document.workspace_name,
                    models.Document.observer,
                    models.Document.observed,
                    models.Document.session_name,
                    models.Document.internal_metadata,
                    models.Document.deleted_at,
                )
            )
            .where(
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
            # Vectors hydrate per chunk; plans only carry ids + content.
            plans.append(
                _CopySpec(
                    observed=source.observed,
                    source_id=source.id,
                    content=source.content,
                    embedding=None,
                    internal_metadata=dict(source.internal_metadata),
                    times_derived=source.times_derived,
                    source_ids=list(source.source_ids)
                    if source.source_ids is not None
                    else None,
                    source_message_ids=source.source_message_ids,
                    session_name=session_name,
                    restore_document_id=soft_deleted_copies.get(key),
                )
            )

    if not plans:
        return 0, set()

    # Phases 2-4 run per chunk so only one chunk's embeddings are alive at a
    # time; each chunk's vectors are dropped once synced.
    store_in_postgres = _store_embeddings_in_postgres()
    touched_observed: set[str] = set()
    copied = 0
    for start in range(0, len(plans), BACKFILL_CHUNK_SIZE):
        chunk = plans[start : start + BACKFILL_CHUNK_SIZE]
        if not await _copy_chunk(
            workspace_name,
            scope_peer,
            session_name,
            chunk,
            store_in_postgres,
            barrier=barrier,
        ):
            return None
        copied += sum(spec.written for spec in chunk)
        touched_observed.update(spec.observed for spec in chunk if spec.written)
        for spec in chunk:
            spec.embedding = None

    return copied, touched_observed


async def _hydrate_chunk_embeddings(
    workspace_name: str, plans: list[_CopySpec]
) -> None:
    """Load this chunk's source embeddings from postgres (if any)."""
    source_ids = [spec.source_id for spec in plans]
    async with tracked_db("scope_backfill.hydrate_embeddings") as db:
        result = await db.execute(
            select(models.Document.id, models.Document.embedding).where(
                models.Document.workspace_name == workspace_name,
                models.Document.id.in_(source_ids),
            )
        )
        by_id = {row.id: _embedding_as_list(row.embedding) for row in result.all()}
    for spec in plans:
        spec.embedding = by_id.get(spec.source_id)


async def _copy_chunk(
    workspace_name: str,
    scope_peer: str,
    session_name: str,
    plans: list[_CopySpec],
    store_in_postgres: bool,
    *,
    barrier: WriteSnapshot,
) -> bool:
    """Embed, write, and sync one chunk. False if the session left the scope.

    The write runs behind the lifecycle barrier with ``barrier``, what the plan
    captured before it read the source documents.

    Raises:
        LifecycleBarrierError: The session, the scope or a sender was deleted,
            or a planned source was erased, since the plan. Nothing is written
    """
    # Phase 2a (DB): pull this chunk's embeddings only.
    await _hydrate_chunk_embeddings(workspace_name, plans)

    # Phase 2b (no DB): fill missing embeddings. Source rows have NULL
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

    # Phase 3 (DB): write the copies, behind the lifecycle barrier. Nothing in
    # this transaction commits before the end: a commit would release the
    # barrier's locks and the membership lock ahead of the inserts.
    touched_observed = {spec.observed for spec in plans}
    chunk_barrier = barrier.for_representations(
        [
            pair
            for observed in touched_observed
            for pair in (
                (observed, observed),
                (scope_peer, observed),
            )
        ],
        peer_names=[scope_peer],
    )
    new_rows: list[models.Document] = []
    new_specs: list[_CopySpec] = []
    async with tracked_db("scope_backfill.write") as db:
        try:
            left_the_scope = False
            async with final_write(db, chunk_barrier):
                # Row-lock active membership for this txn so a concurrent leave
                # (``left_at``) cannot commit between the check and the inserts.
                # After the barrier's parent locks: session deletion locks the
                # session first and the membership rows after it.
                membership = await db.scalar(
                    select(models.SessionPeer.peer_name)
                    .where(models.SessionPeer.tenant_id == models.scoped_tenant_id())
                    .where(models.SessionPeer.workspace_name == workspace_name)
                    .where(models.SessionPeer.session_name == session_name)
                    .where(models.SessionPeer.peer_name == scope_peer)
                    .where(models.SessionPeer.left_at.is_(None))
                    .with_for_update()
                    .limit(1)
                )
                left_the_scope = membership is None
                if not left_the_scope:
                    # Not get_or_create_collection: it commits.
                    await ensure_collections(
                        db,
                        chunk_barrier,
                        [(scope_peer, observed) for observed in touched_observed],
                    )

                    # A source a dream consolidated away since the plan is not
                    # copied. One that was erased also raised its collection's
                    # generation, which fails the whole chunk at the block's
                    # exit, so a restore never brings an erased copy back.
                    live_sources = set(
                        await db.scalars(
                            select(models.Document.id).where(
                                models.Document.tenant_id == chunk_barrier.tenant_id,
                                models.Document.workspace_name == workspace_name,
                                models.Document.id.in_(
                                    [spec.source_id for spec in plans]
                                ),
                                models.Document.deleted_at.is_(None),
                            )
                        )
                    )

                    restore_ids: list[str] = []
                    restored_citations: list[dict[str, Any]] = []
                    for spec in plans:
                        if spec.source_id not in live_sources:
                            continue
                        spec.written = True
                        if spec.restore_document_id is not None:
                            restore_ids.append(spec.restore_document_id)
                            restored_citations.extend(
                                {
                                    "conclusion_id": spec.restore_document_id,
                                    "message_id": link.message_id,
                                    "position": link.position,
                                    "workspace_name": workspace_name,
                                }
                                for link in build_source_message_links(
                                    spec.source_message_ids, workspace_name
                                )
                            )
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
                            source_message_ids=spec.source_message_ids,
                            embedding=spec.embedding if store_in_postgres else None,
                        )
                        row.sync_state = "pending"
                        new_rows.append(row)
                        new_specs.append(spec)

                    db.add_all(new_rows)
                    if restore_ids:
                        await db.execute(
                            update(models.Document)
                            .where(models.Document.id.in_(restore_ids))
                            .values(deleted_at=None, sync_state="pending")
                        )
                        # The source may have gained citations since this copy
                        # was removed; a restored copy takes the source's
                        # current set.
                        await db.execute(
                            delete(models.ConclusionSourceMessage).where(
                                models.ConclusionSourceMessage.conclusion_id.in_(
                                    restore_ids
                                )
                            )
                        )
                        if restored_citations:
                            await db.execute(
                                insert(models.ConclusionSourceMessage),
                                restored_citations,
                            )
            if left_the_scope:
                await db.rollback()
                return False
            # After the block: it locked the collection rows this updates.
            await crud.record_explicit_documents(db, new_rows)
            await db.commit()
        except LifecycleBarrierError:
            await db.rollback()
            for spec in plans:
                spec.written = False
            raise

        # IDs are generated client-side and remain accessible post-commit
        # (expire_on_commit=False), mirroring crud.document.create_documents.
        for spec, row in zip(new_specs, new_rows, strict=True):
            spec.restore_document_id = row.id

    copied_ids = [
        spec.restore_document_id
        for spec in plans
        if spec.written and spec.restore_document_id is not None
    ]

    # Phase 4: sync to the external vector store (or mark synced in pgvector
    # mode). Failures leave rows in sync_state='pending' for the reconciler.
    if copied_ids:
        await _sync_copies_to_vector_store(
            workspace_name,
            scope_peer,
            [spec for spec in plans if spec.written],
            copied_ids,
        )
    return True


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
        namespace = await external_vector_store.get_vector_namespace(
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

    Skipped entirely if the session is an active member again: it was re-added
    after this task was queued, and its copies, notes and status are live data
    the re-add's backfill reconciles.
    """
    scope_peer = payload.scope_peer
    session_name = payload.session_name

    removed_by_observed: dict[str, list[str]] = {}
    async with tracked_db("scope_removal") as db:
        # Row-lock the membership for this whole transaction, as the backfill's
        # write phase does, so a re-add can't commit between this check and the
        # writes below. Nothing here commits until the end, so the lock holds.
        rejoined = await db.scalar(
            select(models.SessionPeer.left_at.is_(None))
            .where(models.SessionPeer.tenant_id == models.scoped_tenant_id())
            .where(models.SessionPeer.workspace_name == workspace_name)
            .where(models.SessionPeer.session_name == session_name)
            .where(models.SessionPeer.peer_name == scope_peer)
            .with_for_update()
        )
        if rejoined:
            logger.info(
                "Scope removal skipped for %s/%s/%s: the session is a member again",
                workspace_name,
                scope_peer,
                session_name,
            )
            return

        observed_result = await db.execute(
            select(models.Collection.observed)
            .where(
                models.Collection.workspace_name == workspace_name,
                models.Collection.observer == scope_peer,
            )
            .order_by(models.Collection.observed)
        )
        observed_in_scope = [row[0] for row in observed_result.all()]
        for observed in observed_in_scope:
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
            # (document_sources) intersects anything removed, transitively — a
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
                        or_(
                            # An entity column, not a bare exists(): then the
                            # tenant predicate hook adds tenant_id to it.
                            select(models.DocumentSource.derived_id)
                            .where(
                                models.DocumentSource.tenant_id
                                == models.Document.tenant_id,
                                models.DocumentSource.derived_id == models.Document.id,
                                models.DocumentSource.source_id.in_(frontier),
                                models.DocumentSource.workspace_name == workspace_name,
                            )
                            .exists(),
                            # Undrained rows still link through the legacy column.
                            models.Document.legacy_source_ids.has_any(array(frontier)),
                        ),
                    )
                    .values(deleted_at=func.now())
                    .returning(models.Document.id)
                )
                frontier = [row[0] for row in (await db.execute(derived_stmt)).all()]
                all_removed.extend(frontier)

            if all_removed:
                removed_by_observed[observed] = all_removed

        # Dream notes are free text, possibly written from the removed
        # session's messages even where it left no conclusions, so they go for
        # every collection in the scope, in the same commit as the conclusions.
        # Unlike the card they can't be rebuilt from what remains. A new epoch
        # turns away note writes from dreams that started before this, which
        # may still hold the old notes. The cursor stays.
        epoch = str(generate_nanoid())
        for observed in observed_in_scope:
            await crud.merge_collection_internal_metadata_value(
                db,
                workspace_name,
                scope_peer,
                observed,
                NOTES_KEY,
                {"entries": [], EPOCH_KEY: epoch},
                commit=False,
            )

        # A dream that read the removed copies must not write from them. The
        # rows are already locked by the notes write above.
        await bump_collection_generations(
            db,
            workspace_name,
            [(scope_peer, observed) for observed in removed_by_observed],
        )

        # The session left the scope, so its backfill status entry is moot;
        # clearing it keeps a later re-add starting from a fresh "pending".
        await crud.clear_scope_backfill_status(
            db, workspace_name, scope_peer, session_name
        )
        await db.commit()
    await crud.invalidate_scope_peer_cache(workspace_name, scope_peer)
    for observed in observed_in_scope:
        await crud.invalidate_collection_cache(workspace_name, scope_peer, observed)

    # Delete the vectors eagerly so recall can't surface removed memory while
    # waiting for the reconciler sweep (which remains the backstop on failure).
    external_vector_store = get_external_vector_store()
    if external_vector_store is not None:
        for observed, removed_ids in removed_by_observed.items():
            namespace = await external_vector_store.get_vector_namespace(
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
