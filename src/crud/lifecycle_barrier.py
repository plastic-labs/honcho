"""The deletion lifecycle barrier: the one final-write step every producer runs.

A producer (deriver, dreamer, summarizer, card writer, scope backfill, enqueue,
a public write route) reads its input, often calls a model, and only then
writes. Removing its queued work does not stop that write. The barrier does:

1. **Capture.** Before it reads its input, the producer calls
   ``capture_write_snapshot``. The snapshot holds the tenant-qualified stable
   ids of the workspace, session, peers and collections it will write under,
   and the ``deletion_generation`` of each row that has one. It never holds a
   database lock, so it is safe to carry across a model call.
2. **Final write.** The producer opens its write transaction and runs its
   writes inside ``final_write(db, snapshot)``. That step never commits; the
   producer owns the transaction and commits after it.

``final_write`` takes its locks in one global order:

    workspace -> peers (id order) -> session -> [the producer's document
    writes] -> collections (sorted key order) -> queue -> active_queue_sessions

Parents are locked first, before any document row, and compared with the
snapshot. The collection check runs after the document writes, which keeps the
"documents before collections" rule. The queue tables come last, in the order a
completing worker and a deletion take them (queue, then the trigger-maintained
``queue_item_batches``, then ``active_queue_sessions``).

**What a deleter must do.** The barrier only serializes against a deletion that
follows this contract, in its own transaction:

- lock the row it deletes or tombstones with ``FOR UPDATE`` *before* it removes
  anything under it, and raise that row's ``deletion_generation`` when it has
  one (``begin_workspace_deletion``, ``begin_session_deletion``);
- raise the ``deletion_generation`` of every collection it removes conclusions
  from, after it wrote the documents (``bump_collection_generations``);
- when it deletes something that has no generation of its own (a peer, a
  message), raise the generation of each row a producer captured in its place:
  the session a message belongs to, and every session and collection a peer is
  a member of, observes, or is observed in.

Producers lock parents ``FOR KEY SHARE``, the lock a foreign-key check already
takes, so they do not block each other or ordinary updates of those rows. It
conflicts only with ``FOR UPDATE``, which is why the deleter must take that.

Generations are compared, never names: a session deleted and recreated under
the same name has a new id and does not pass. Only ``deletion_generation`` is
compared here. A change counter for every write is a separate concern.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

from sqlalchemy import select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.lifecycle import live_message_row
from src.exceptions import (
    WriteClaimLostError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)

__all__ = [
    "ClaimStamp",
    "CollectionStamp",
    "GenerationStamp",
    "PeerStamp",
    "WriteSnapshot",
    "begin_session_deletion",
    "begin_workspace_deletion",
    "bump_collection_generations",
    "capture_write_snapshot",
    "claim_context",
    "current_claim",
    "ensure_collections",
    "final_write",
    "stamp_representations",
]


@dataclass(frozen=True, slots=True)
class GenerationStamp:
    """A workspace or session row: its stable id, name and captured generation."""

    uid: str
    name: str
    deletion_generation: int


@dataclass(frozen=True, slots=True)
class PeerStamp:
    """A peer row. Peers have no generation, so the barrier checks liveness by id."""

    uid: str
    name: str


@dataclass(frozen=True, slots=True)
class CollectionStamp:
    """One representation, ``(observer, observed)``, and its captured generation.

    ``uid`` is None when the collection did not exist at capture. A collection
    starts at generation 0, so one created since then passes only at 0.
    """

    observer: str
    observed: str
    uid: str | None
    deletion_generation: int


@dataclass(frozen=True, slots=True)
class ClaimStamp:
    """The claim a queue worker holds on its work unit, and the rows it took.

    ``tenant_id`` is the value stored on the claim and queue rows, which is NULL
    when ``MULTI_TENANT`` is off and for tenant-less work. It is matched with
    ``IS NOT DISTINCT FROM``, together with the claim's own id, so a work-unit
    key that reads the same for two tenants never matches the other's rows.
    """

    work_unit_key: str
    claim_id: str
    tenant_id: str | None
    queue_item_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class WriteSnapshot:
    """What a producer captured before it read its input. Holds no locks.

    ``message_ids`` are ``Message.public_id`` values of every source message
    the output depends on, including context-only messages that have no queue
    row of their own.
    """

    tenant_id: str
    workspace: GenerationStamp
    session: GenerationStamp | None = None
    peers: tuple[PeerStamp, ...] = ()
    collections: tuple[CollectionStamp, ...] = ()
    message_ids: tuple[str, ...] = ()
    claim: ClaimStamp | None = None

    def with_messages(self, message_ids: Iterable[str]) -> WriteSnapshot:
        """Add source messages read after the capture."""
        merged = dict.fromkeys((*self.message_ids, *message_ids))
        return replace(self, message_ids=tuple(merged))

    def with_claim(self, claim: ClaimStamp | None) -> WriteSnapshot:
        return replace(self, claim=claim)

    def for_representation(self, observer: str, observed: str) -> WriteSnapshot:
        """Narrow to one representation, for a write that touches only it.

        Keeps the workspace, session, messages and claim. Keeps only the two
        peers and the one collection, so a write to one observer's collection
        does not lock, or fail on, another observer's.
        """
        return replace(
            self,
            peers=tuple(p for p in self.peers if p.name in (observer, observed)),
            collections=tuple(
                c
                for c in self.collections
                if (c.observer, c.observed) == (observer, observed)
            ),
        )

    def peer(self, name: str) -> PeerStamp | None:
        return next((p for p in self.peers if p.name == name), None)


# The claim of the work unit the current task is processing, set by the queue
# worker for the duration of the unit. Producers that run off the queue read it
# at capture so their final write also proves the worker still holds its unit.
_claim: ContextVar[ClaimStamp | None] = ContextVar("lifecycle_claim", default=None)


def current_claim(queue_item_ids: Iterable[int] = ()) -> ClaimStamp | None:
    """The running work unit's claim, narrowed to the queue rows in hand."""
    claim = _claim.get()
    if claim is None:
        return None
    return replace(claim, queue_item_ids=tuple(sorted(set(queue_item_ids))))


@asynccontextmanager
async def claim_context(claim: ClaimStamp | None) -> AsyncGenerator[None]:
    """Bind the work unit's claim for the producers that run inside the block."""
    token = _claim.set(claim)
    try:
        yield
    finally:
        _claim.reset(token)


def _tenant(tenant_id: str | None) -> str:
    tenant = tenant_id or (
        models.scoped_tenant_id()
        if settings.MULTI_TENANT
        else models.instance_tenant_id()
    )
    if not tenant:
        raise ValueError("A write snapshot requires a tenant_id")
    return tenant


async def capture_write_snapshot(
    db: AsyncSession,
    workspace_name: str,
    *,
    tenant_id: str | None = None,
    session_name: str | None = None,
    peer_names: Iterable[str] = (),
    representations: Iterable[tuple[str, str]] = (),
    message_ids: Iterable[str] = (),
    claim: ClaimStamp | None = None,
) -> WriteSnapshot:
    """Resolve names to tenant-qualified ids and read the current generations.

    Call it **before** reading the input the write is derived from. A deletion
    that commits after this call is then caught by ``final_write``; one that
    committed before it is already absent from the input read afterwards.
    Calling it after the input read leaves a window in which a deletion is
    neither seen in the input nor caught at the write.

    Plain reads: no lock, flush, or commit. Every predicate carries the tenant,
    so it is correct on a tenant-bound session and on a service session.

    Args:
        db: The caller's session
        workspace_name: The workspace the write lands in
        tenant_id: The tenant. Defaults to the bound tenant under
            ``MULTI_TENANT`` and to the instance tenant otherwise
        session_name: The session the write lands in or reads from, if any
        peer_names: Peers that must stay live. The peers of ``representations``
            are added
        representations: ``(observer, observed)`` pairs the write touches
        message_ids: Public ids of source messages already known
        claim: The work unit's claim, when the producer runs off the queue

    Raises:
        WriteParentGoneError: The workspace, session or a peer does not exist
            or is tombstoned
    """
    tenant = _tenant(tenant_id)
    pairs = sorted(set(representations))
    names = sorted({*peer_names, *(name for pair in pairs for name in pair)})

    workspace_row = (
        await db.execute(
            select(models.Workspace.id, models.Workspace.deletion_generation).where(
                models.Workspace.tenant_id == tenant,
                models.Workspace.name == workspace_name,
                models.Workspace.deleted_at.is_(None),
            )
        )
    ).one_or_none()
    if workspace_row is None:
        raise WriteParentGoneError("workspace", f"Workspace {workspace_name} is gone")
    workspace = GenerationStamp(
        workspace_row.id, workspace_name, workspace_row.deletion_generation
    )

    session: GenerationStamp | None = None
    if session_name is not None:
        session_row = (
            await db.execute(
                select(models.Session.id, models.Session.deletion_generation).where(
                    models.Session.tenant_id == tenant,
                    models.Session.workspace_name == workspace_name,
                    models.Session.name == session_name,
                    models.Session.deleted_at.is_(None),
                )
            )
        ).one_or_none()
        if session_row is None:
            raise WriteParentGoneError("session", f"Session {session_name} is gone")
        session = GenerationStamp(
            session_row.id, session_name, session_row.deletion_generation
        )

    snapshot = WriteSnapshot(
        tenant_id=tenant,
        workspace=workspace,
        session=session,
        message_ids=tuple(dict.fromkeys(message_ids)),
        claim=claim,
    )
    if not names:
        return snapshot
    return await stamp_representations(db, snapshot, pairs, peer_names=names)


async def stamp_representations(
    db: AsyncSession,
    snapshot: WriteSnapshot,
    representations: Iterable[tuple[str, str]],
    *,
    peer_names: Iterable[str] = (),
) -> WriteSnapshot:
    """Add peers and ``(observer, observed)`` collections to a snapshot.

    For a producer that learns which representations it writes only from its
    input. Call it before reading anything from those collections. The
    workspace and session generations captured earlier still cover the input
    read in between, given the deleter contract in the module docstring.

    Raises:
        WriteParentGoneError: A peer does not exist or is tombstoned
    """
    pairs = sorted(set(representations))
    names = sorted({*peer_names, *(name for pair in pairs for name in pair)})
    known = {p.name: p for p in snapshot.peers}
    missing = [name for name in names if name not in known]
    if missing:
        rows = await db.execute(
            select(models.Peer.id, models.Peer.name).where(
                models.Peer.tenant_id == snapshot.tenant_id,
                models.Peer.workspace_name == snapshot.workspace.name,
                models.Peer.name.in_(missing),
                models.Peer.deleted_at.is_(None),
            )
        )
        for row in rows:
            known[row.name] = PeerStamp(row.id, row.name)
        for name in missing:
            if name not in known:
                raise WriteParentGoneError("peer", f"Peer {name} is gone")

    stamped = {(c.observer, c.observed): c for c in snapshot.collections}
    new_pairs = [pair for pair in pairs if pair not in stamped]
    if new_pairs:
        rows = await db.execute(
            select(
                models.Collection.id,
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.deletion_generation,
            ).where(
                models.Collection.tenant_id == snapshot.tenant_id,
                models.Collection.workspace_name == snapshot.workspace.name,
                tuple_(models.Collection.observer, models.Collection.observed).in_(
                    new_pairs
                ),
            )
        )
        for row in rows:
            stamped[(row.observer, row.observed)] = CollectionStamp(
                row.observer, row.observed, row.id, row.deletion_generation
            )
        for observer, observed in new_pairs:
            stamped.setdefault(
                (observer, observed), CollectionStamp(observer, observed, None, 0)
            )

    return replace(
        snapshot,
        peers=tuple(known[name] for name in sorted(known)),
        collections=tuple(stamped[pair] for pair in sorted(stamped)),
    )


@asynccontextmanager
async def final_write(
    db: AsyncSession, snapshot: WriteSnapshot
) -> AsyncGenerator[None]:
    """Run a producer's writes behind the barrier, in the producer's transaction.

    On entry it locks the workspace, peers and session ``FOR KEY SHARE`` and
    compares each with the snapshot, then checks the source messages. The
    caller's writes go in the block. On a clean exit it flushes those writes,
    locks the snapshot's collections in sorted key order and compares their
    generations, then checks the worker's queue rows and claim.

    It never commits and never rolls back: the caller does both. Nothing may
    commit inside the block, or the locks are released before the write.
    Statements that change the snapshot's collection rows belong after the
    block, so that those rows are first locked here, in order.

    Raises:
        WriteParentGoneError: A parent or source was deleted. Drop the work
        WriteGenerationStaleError: A deletion changed the input. Recompute
        WriteClaimLostError: The worker no longer holds its work unit
    """
    await _lock_parents(db, snapshot)
    yield
    await db.flush()
    await _check_collections(db, snapshot)
    await _check_claim(db, snapshot)


async def _lock_parents(db: AsyncSession, snapshot: WriteSnapshot) -> None:
    tenant = snapshot.tenant_id
    workspace = (
        await db.execute(
            select(
                models.Workspace.name,
                models.Workspace.deleted_at,
                models.Workspace.deletion_generation,
            )
            .where(
                models.Workspace.tenant_id == tenant,
                models.Workspace.id == snapshot.workspace.uid,
            )
            .with_for_update(read=True, key_share=True)
        )
    ).one_or_none()
    if (
        workspace is None
        or workspace.deleted_at is not None
        or workspace.name != snapshot.workspace.name
    ):
        raise WriteParentGoneError(
            "workspace", f"Workspace {snapshot.workspace.name} was deleted"
        )
    if workspace.deletion_generation != snapshot.workspace.deletion_generation:
        raise WriteGenerationStaleError(
            "workspace",
            f"Workspace {snapshot.workspace.name} had a deletion since capture",
        )

    if snapshot.peers:
        # Explicit id order: an IN list does not order the locks.
        rows = await db.execute(
            select(models.Peer.id, models.Peer.name, models.Peer.deleted_at)
            .where(
                models.Peer.tenant_id == tenant,
                models.Peer.workspace_name == snapshot.workspace.name,
                models.Peer.id.in_([p.uid for p in snapshot.peers]),
            )
            .order_by(models.Peer.id)
            .with_for_update(read=True, key_share=True)
        )
        live = {row.id: row.name for row in rows if row.deleted_at is None}
        for peer in snapshot.peers:
            if live.get(peer.uid) != peer.name:
                raise WriteParentGoneError("peer", f"Peer {peer.name} was deleted")

    if snapshot.session is not None:
        session = (
            await db.execute(
                select(
                    models.Session.name,
                    models.Session.deleted_at,
                    models.Session.deletion_generation,
                )
                .where(
                    models.Session.tenant_id == tenant,
                    models.Session.workspace_name == snapshot.workspace.name,
                    models.Session.id == snapshot.session.uid,
                )
                .with_for_update(read=True, key_share=True)
            )
        ).one_or_none()
        if (
            session is None
            or session.deleted_at is not None
            or session.name != snapshot.session.name
        ):
            raise WriteParentGoneError(
                "session", f"Session {snapshot.session.name} was deleted"
            )
        if session.deletion_generation != snapshot.session.deletion_generation:
            raise WriteGenerationStaleError(
                "session",
                f"Session {snapshot.session.name} had a deletion since capture",
            )

    if snapshot.message_ids:
        # No lock: a message deletion raises its session's generation under
        # FOR UPDATE, which the session lock above already waited for.
        live_ids = set(
            await db.scalars(
                select(models.Message.public_id).where(
                    models.Message.tenant_id == tenant,
                    models.Message.workspace_name == snapshot.workspace.name,
                    models.Message.public_id.in_(snapshot.message_ids),
                    live_message_row(),
                )
            )
        )
        if len(live_ids) != len(set(snapshot.message_ids)):
            raise WriteGenerationStaleError(
                "message", "A source message was deleted since capture"
            )


async def _check_collections(db: AsyncSession, snapshot: WriteSnapshot) -> None:
    # One row per statement, in the Python-sorted key order that
    # crud.collection.record_explicit_documents updates them in. A single
    # ORDER BY would sort by the database collation, which can differ.
    for stamp in sorted(snapshot.collections, key=lambda c: (c.observer, c.observed)):
        row = (
            await db.execute(
                select(models.Collection.id, models.Collection.deletion_generation)
                .where(
                    models.Collection.tenant_id == snapshot.tenant_id,
                    models.Collection.workspace_name == snapshot.workspace.name,
                    models.Collection.observer == stamp.observer,
                    models.Collection.observed == stamp.observed,
                )
                .with_for_update(key_share=True)
            )
        ).one_or_none()
        if row is None:
            if stamp.uid is None:
                # Not there at capture and not created by this write.
                continue
            raise WriteParentGoneError(
                "collection",
                f"Collection {stamp.observer}/{stamp.observed} was deleted",
            )
        if stamp.uid is not None and row.id != stamp.uid:
            raise WriteParentGoneError(
                "collection",
                f"Collection {stamp.observer}/{stamp.observed} was deleted",
            )
        if row.deletion_generation != stamp.deletion_generation:
            raise WriteGenerationStaleError(
                "collection",
                f"Collection {stamp.observer}/{stamp.observed} had a deletion "
                + "since capture",
            )


async def _check_claim(db: AsyncSession, snapshot: WriteSnapshot) -> None:
    claim = snapshot.claim
    if claim is None:
        return
    if claim.queue_item_ids:
        # Queue rows first and in id order, then the claim row: the order a
        # completing worker and a deletion take them in.
        held = (
            await db.scalars(
                select(models.QueueItem.id)
                .where(
                    models.QueueItem.id.in_(claim.queue_item_ids),
                    models.QueueItem.work_unit_key == claim.work_unit_key,
                    models.QueueItem.tenant_id.is_not_distinct_from(claim.tenant_id),
                    models.QueueItem.processed.is_(False),
                )
                .order_by(models.QueueItem.id)
                .with_for_update(read=True, key_share=True)
            )
        ).all()
        if len(held) != len(set(claim.queue_item_ids)):
            raise WriteClaimLostError(
                "claim", f"Queued work for {claim.work_unit_key} was removed"
            )
    claim_row = await db.scalar(
        select(models.ActiveQueueSession.id)
        .where(
            models.ActiveQueueSession.id == claim.claim_id,
            models.ActiveQueueSession.work_unit_key == claim.work_unit_key,
            models.ActiveQueueSession.tenant_id.is_not_distinct_from(claim.tenant_id),
        )
        .with_for_update(read=True, key_share=True)
    )
    if claim_row is None:
        raise WriteClaimLostError(
            "claim", f"The claim on {claim.work_unit_key} was revoked"
        )


async def ensure_collections(
    db: AsyncSession, snapshot: WriteSnapshot, pairs: Iterable[tuple[str, str]]
) -> None:
    """Create the missing collections of ``pairs`` without committing.

    For use inside ``final_write``: ``crud.get_or_create_collection`` commits,
    which would release the barrier's locks before the write.
    """
    rows = [
        {
            "tenant_id": snapshot.tenant_id,
            "workspace_name": snapshot.workspace.name,
            "observer": observer,
            "observed": observed,
        }
        for observer, observed in sorted(set(pairs))
    ]
    if not rows:
        return
    await db.execute(
        insert(models.Collection)
        .values(rows)
        .on_conflict_do_nothing(
            index_elements=["tenant_id", "observer", "observed", "workspace_name"]
        )
    )


# ---------------------------------------------------------------------------
# The deleter's side of the contract
# ---------------------------------------------------------------------------


async def begin_workspace_deletion(
    db: AsyncSession, workspace_name: str, *, tenant_id: str | None = None
) -> str | None:
    """Lock a workspace ``FOR UPDATE`` and raise its generation. First statement
    of a workspace deletion. Returns the workspace id, or None if it is absent.
    """
    tenant = _tenant(tenant_id)
    uid = await db.scalar(
        select(models.Workspace.id)
        .where(
            models.Workspace.tenant_id == tenant,
            models.Workspace.name == workspace_name,
        )
        .with_for_update()
    )
    if uid is None:
        return None
    await db.execute(
        update(models.Workspace)
        .where(models.Workspace.tenant_id == tenant, models.Workspace.id == uid)
        .values(deletion_generation=models.Workspace.deletion_generation + 1)
    )
    return uid


async def begin_session_deletion(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    *,
    tenant_id: str | None = None,
) -> str | None:
    """Lock a session ``FOR UPDATE`` and raise its generation. First statement
    of a session deletion, and of any deletion of that session's messages.
    Returns the session id, or None if it is absent.
    """
    tenant = _tenant(tenant_id)
    uid = await db.scalar(
        select(models.Session.id)
        .where(
            models.Session.tenant_id == tenant,
            models.Session.workspace_name == workspace_name,
            models.Session.name == session_name,
        )
        .with_for_update()
    )
    if uid is None:
        return None
    await db.execute(
        update(models.Session)
        .where(models.Session.tenant_id == tenant, models.Session.id == uid)
        .values(deletion_generation=models.Session.deletion_generation + 1)
    )
    return uid


async def bump_collection_generations(
    db: AsyncSession,
    workspace_name: str,
    pairs: Iterable[tuple[str, str]],
    *,
    tenant_id: str | None = None,
) -> None:
    """Raise the generation of each ``(observer, observed)`` collection.

    A deleter calls it after it wrote the documents it removes, never before:
    documents are locked before collections. Rows are taken one at a time in
    sorted key order, the order every other writer locks them in.
    """
    tenant = _tenant(tenant_id)
    for observer, observed in sorted(set(pairs)):
        await db.execute(
            update(models.Collection)
            .where(
                models.Collection.tenant_id == tenant,
                models.Collection.workspace_name == workspace_name,
                models.Collection.observer == observer,
                models.Collection.observed == observed,
            )
            .values(deletion_generation=models.Collection.deletion_generation + 1)
        )
