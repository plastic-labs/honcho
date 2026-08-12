"""CRUD helpers for scopes.

A scope is a named grouping of sessions, implemented as a peer named
``scope.<name>`` carrying ``{"kind": "scope"}`` in ``internal_metadata`` (the
authoritative, user-unwritable flag) and ``{"observe_me": false}`` in
``configuration``, that observes its member sessions (``observe_others=true``)
and never speaks.
See ``src/utils/scopes.py`` for the namespace helpers.

Messages ingested after a membership change flow to the scope via the normal
deriver fan-out. Retroactive changes are handled by queue jobs: adding a session
that already has messages enqueues a ``scope_backfill`` task (copy of the
session's explicit documents into the scope's collections) and removing a
session enqueues a ``scope_removal`` task (soft-delete of the session's
documents plus dependent derived documents).
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Literal
from typing import cast as py_cast

from sqlalchemy import Select, Text, cast, select, update
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, array
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import func

from src import models, schemas
from src.cache.client import safe_cache_delete
from src.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)
from src.utils.scopes import (
    SCOPE_KIND,
    is_scope_peer,
    scope_peer_name,
)
from src.utils.types import GetOrCreateResult

from .peer import peer_cache_key, scope_peer_clause
from .workspace import get_or_create_workspace

logger = getLogger(__name__)

# Key inside the scope peer's internal_metadata that holds per-session
# backfill job status: {<session_name>: {state, updated_at[, docs_copied]}}.
BACKFILL_STATUS_KEY = "backfill_status"

ScopeBackfillState = Literal["pending", "completed", "failed"]

# Internal metadata stamped on every scope peer at creation. `kind` is the
# authoritative scope flag and lives here — NOT in `configuration` — because
# `configuration` is user-writable (`PeerCreate`/`PeerUpdate` accept a free-form
# dict, and `update_peer` replaces it wholesale), so a user could forge or clear
# the flag. `internal_metadata` appears in no API schema at all.
#
# Backfill job status shares this field under BACKFILL_STATUS_KEY below. Every
# write to it must be a JSONB merge scoped to that key, never a wholesale
# replacement, or the `kind` flag goes with it and the peer stops being a scope.
SCOPE_PEER_INTERNAL_METADATA: dict[str, str] = {
    "kind": SCOPE_KIND,
}

# Peer-level configuration stamped on every scope peer at creation.
# `observe_me: false` ensures no representation is ever formed *of* a scope peer.
# This one stays user-visible: `observe_me` is a legitimate config knob.
SCOPE_PEER_CONFIGURATION: dict[str, str | bool] = {
    "observe_me": False,
}

# Session-level configuration for a scope peer's membership in a session.
SCOPE_MEMBERSHIP_CONFIG = schemas.SessionPeerConfig(
    observe_others=True, observe_me=False
)


async def get_or_create_scopes(
    db: AsyncSession,
    workspace_name: str,
    scopes: list[schemas.ScopeCreate],
    *,
    _retry: bool = False,
    _pending_invalidation: list[str] | None = None,
) -> GetOrCreateResult[list[models.Peer]]:
    """
    Get existing scopes or create new ones if they don't exist.

    Existing scope peers have their metadata updated when provided. A
    pre-existing peer that occupies a scope's reserved name *without* the
    authoritative ``kind`` flag (a legacy collision) is never adopted.

    Note: does not commit; the caller owns the transaction (mirror of
    ``get_or_create_peers``). Run ``result.post_commit()`` after committing.

    Deliberately does NOT scan for pre-existing state naming the backing peer
    (peer-card keys, pending dream queue items). ``reject_scope_observed`` now
    refuses writes against a not-yet-existing reserved name, so no new such state
    can be created; only data written before that guard existed could collide, and
    since ``scope.`` was never a meaningful namespace then, any such row is
    coincidental. The consequence would also be inert — a card or queue item
    describing a scope, which nothing reads, because no representation is formed of
    a scope. Detecting card keys means scanning every peer's ``internal_metadata``
    for a label containing this name, i.e. a full table scan per scope creation:
    disproportionate to that risk. Revisit if scope names ever become guessable
    across tenants.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scopes: List of scope creation schemas (unprefixed names)
        _retry: Whether this is the retry attempt
        _pending_invalidation: Names of scope peers already mutated by a prior
            attempt, whose cache keys must still be purged. See the retry branch.

    Returns:
        GetOrCreateResult containing the backing peers and whether any were
        created

    Raises:
        ConflictException: If a peer already occupies a scope's reserved name
            without the scope kind flag, or if we fail to get or create the
            scope peers
    """
    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))

    peer_names = {scope_peer_name(s.name): s for s in scopes}
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names.keys()))
    )
    result = await db.execute(stmt)
    existing_peers: list[models.Peer] = list(result.scalars().all())

    changed_peers: list[models.Peer] = []
    for existing_peer in existing_peers:
        if not is_scope_peer(existing_peer.name, existing_peer.internal_metadata):
            raise ConflictException(
                f"A peer named '{existing_peer.name}' already exists in workspace "
                + f"{workspace_name} but is not a scope. Rename or delete that "
                + "peer before creating this scope."
            )
        scope_schema = peer_names[existing_peer.name]
        if (
            scope_schema.metadata is not None
            and existing_peer.h_metadata != scope_schema.metadata
        ):
            existing_peer.h_metadata = scope_schema.metadata
            changed_peers.append(existing_peer)

    existing_names = {p.name for p in existing_peers}
    new_peers = [
        models.Peer(
            workspace_name=workspace_name,
            name=name,
            h_metadata=scope_schema.metadata or {},
            internal_metadata=dict(SCOPE_PEER_INTERNAL_METADATA),
            configuration=dict(SCOPE_PEER_CONFIGURATION),
        )
        for name, scope_schema in peer_names.items()
        if name not in existing_names
    ]
    try:
        async with db.begin_nested():
            db.add_all(new_peers)
    except IntegrityError:
        if _retry:
            raise ConflictException(
                f"Unable to create or get scopes: {sorted(peer_names)}"
            ) from None
        # `begin_nested()` autoflushes the mutations above *before* opening the
        # savepoint, so they survive the rollback and leave the ORM state clean —
        # the retry would compare already-updated values, find no change, and skip
        # the purge. Carry the names forward so the invalidation can't be lost.
        return await get_or_create_scopes(
            db,
            workspace_name,
            scopes,
            _retry=True,
            _pending_invalidation=(_pending_invalidation or [])
            + [p.name for p in changed_peers],
        )

    _cache_keys_to_invalidate = [
        peer_cache_key(workspace_name, name)
        for name in dict.fromkeys(
            (_pending_invalidation or []) + [p.name for p in changed_peers + new_peers]
        )
    ]

    async def _invalidate_peer_cache():
        for cache_key in _cache_keys_to_invalidate:
            await safe_cache_delete(cache_key)

    return GetOrCreateResult(
        existing_peers + new_peers,
        created=len(new_peers) > 0,
        on_commit=_invalidate_peer_cache if _cache_keys_to_invalidate else None,
    )


async def get_scopes(
    workspace_name: str,
    reverse: bool = False,
) -> Select[tuple[models.Peer]]:
    """Build a scope list query, ordered by creation time.

    Requires both halves via ``scope_peer_clause`` (reserved name prefix AND the
    internal kind flag), so a peer carrying a forged ``configuration`` cannot
    inject itself into the scope list.
    """
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(scope_peer_clause())
    )
    if reverse:
        return stmt.order_by(models.Peer.created_at.desc(), models.Peer.id.desc())
    return stmt.order_by(models.Peer.created_at.asc(), models.Peer.id.asc())


async def get_scope_or_raise(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
) -> models.Peer:
    """
    Get an existing scope's backing peer by its unprefixed scope name.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name

    Returns:
        The backing peer if found and flagged as a scope

    Raises:
        ResourceNotFoundException: If no scope with that name exists (a peer
            occupying the reserved name without the kind flag does not count)
    """
    peer = await db.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == scope_peer_name(scope_name))
    )
    if peer is None or not is_scope_peer(peer.name, peer.internal_metadata):
        raise ResourceNotFoundException(
            f"Scope {scope_name} not found in workspace {workspace_name}"
        )
    return peer


async def resolve_scope_peers(
    db: AsyncSession,
    workspace_name: str,
    scope_names: Sequence[str],
) -> list[str]:
    """
    Resolve unprefixed scope names to their backing scope-peer names.

    Used by the read routes that accept a ``scope`` option (chat,
    representation, session context, workspace search) to turn user-facing
    scope names into the observer peers that implement them.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_names: Unprefixed scope names (duplicates are collapsed,
            preserving first-seen order)

    Returns:
        The backing scope-peer names, in first-requested order

    Raises:
        ResourceNotFoundException: If any named scope does not exist
        ValidationException: If a peer occupies a scope's reserved name
            without the authoritative kind flag (a legacy collision)
    """
    requested: list[str] = []
    seen: set[str] = set()
    for name in scope_names:
        if name not in seen:
            seen.add(name)
            requested.append(name)

    peer_names = [scope_peer_name(name) for name in requested]
    if not peer_names:
        return []

    result = await db.execute(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names))
    )
    peers_by_name = {peer.name: peer for peer in result.scalars().all()}

    resolved: list[str] = []
    for name, peer_name in zip(requested, peer_names, strict=True):
        peer = peers_by_name.get(peer_name)
        if peer is None:
            raise ResourceNotFoundException(
                f"Scope {name} not found in workspace {workspace_name}"
            )
        # The kind flag is authoritative and lives in internal_metadata, so a
        # legacy peer merely occupying the reserved name is refused rather than
        # silently treated as a scope.
        if not is_scope_peer(peer.name, peer.internal_metadata):
            raise ValidationException(
                f"'{name}' does not name a scope: a non-scope peer occupies "
                + "its reserved name."
            )
        resolved.append(peer_name)
    return resolved


async def get_scope_sessions(
    workspace_name: str,
    scope_name: str,
    reverse: bool = False,
) -> Select[tuple[models.Session]]:
    """
    Build a query for the active sessions that are members of a scope.

    Membership is unbounded — a scope may span every session in a workspace — so
    this returns a query for the caller to paginate rather than a materialized
    list. Callers must check the scope exists themselves (``get_scope_or_raise``);
    an unknown scope yields an empty page here, not a 404.

    Ordered by membership age, with the session id as a unique tiebreaker:
    ``session_peers`` has a composite primary key and no id of its own, so
    ``joined_at`` alone is not a stable pagination key.

    Args:
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name
        reverse: Whether to return newest memberships first

    Returns:
        Select for the scope's member sessions
    """
    stmt = (
        select(models.Session)
        .join(
            models.SessionPeer,
            (models.Session.name == models.SessionPeer.session_name)
            & (models.Session.workspace_name == models.SessionPeer.workspace_name),
        )
        .where(models.SessionPeer.workspace_name == workspace_name)
        .where(models.SessionPeer.peer_name == scope_peer_name(scope_name))
        .where(models.SessionPeer.left_at.is_(None))
        .where(models.Session.is_active == True)  # noqa: E712
    )
    if reverse:
        return stmt.order_by(
            models.SessionPeer.joined_at.desc(), models.Session.id.desc()
        )
    return stmt.order_by(models.SessionPeer.joined_at.asc(), models.Session.id.asc())


async def add_sessions_to_scope(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
    session_names: list[str],
) -> None:
    """
    Add sessions to a scope by creating observer memberships for its peer.

    Each membership is a ``session_peers`` row for the scope peer with
    ``observe_others=true, observe_me=false`` — exactly what a hand-built
    observer peer would carry. Sessions that already have messages get a
    ``scope_backfill`` queue task so their existing explicit documents are
    copied into the scope's collections; fresh sessions need nothing.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name
        session_names: Names of existing sessions to add

    Raises:
        ResourceNotFoundException: If the scope or any named session does not
            exist
    """
    # Imported lazily: crud.session imports this module for the session-create
    # `scopes` path, so a module-level import would be circular.
    from .session import upsert_session_peers

    await get_scope_or_raise(db, workspace_name, scope_name)

    requested = set(session_names)
    result = await db.execute(
        select(models.Session.name)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name.in_(requested))
        .where(models.Session.is_active == True)  # noqa: E712
    )
    found = {row[0] for row in result.all()}
    missing = sorted(requested - found)
    if missing:
        raise ResourceNotFoundException(
            f"Session(s) {missing} not found in workspace {workspace_name}"
        )

    for session_name in sorted(requested):
        await upsert_session_peers(
            db,
            workspace_name=workspace_name,
            session_name=session_name,
            peer_names={scope_peer_name(scope_name): SCOPE_MEMBERSHIP_CONFIG},
            fetch_after_upsert=False,
        )

    await db.commit()

    # Backfill: only sessions that already have messages need it.
    # Imported lazily: src.deriver.enqueue imports crud at module level.
    from src.deriver.enqueue import enqueue_scope_backfill

    msg_result = await db.execute(
        select(models.Message.session_name)
        .where(models.Message.workspace_name == workspace_name)
        .where(models.Message.session_name.in_(requested))
        .distinct()
    )
    for session_with_messages in sorted({row[0] for row in msg_result.all()}):
        await enqueue_scope_backfill(
            workspace_name,
            scope_peer=scope_peer_name(scope_name),
            session_name=session_with_messages,
        )


async def remove_session_from_scope(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
    session_name: str,
) -> None:
    """
    Remove a session from a scope by ending the scope peer's membership.

    Ends the membership the same way the generic remove-peer path does (sets
    ``left_at``), then enqueues a ``scope_removal`` reconciliation task that
    soft-deletes the session's documents from the scope's collections along
    with any derived documents whose support left the scope.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name
        session_name: Name of the session to remove

    Raises:
        ResourceNotFoundException: If the scope or session does not exist
    """
    # Lazy import for the same circular-import reason as add_sessions_to_scope.
    from src.deriver.enqueue import enqueue_scope_removal

    from .session import remove_peers_from_session

    await get_scope_or_raise(db, workspace_name, scope_name)

    await remove_peers_from_session(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        peer_names={scope_peer_name(scope_name)},
        # This *is* the supported path for ending scope membership.
        _allow_scope_peers=True,
    )

    await enqueue_scope_removal(
        workspace_name,
        scope_peer=scope_peer_name(scope_name),
        session_name=session_name,
    )


async def update_scope_backfill_status(
    db: AsyncSession,
    workspace_name: str,
    scope_peer: str,
    session_name: str,
    *,
    state: ScopeBackfillState,
    docs_copied: int | None = None,
) -> None:
    """
    Merge one session's backfill status into the scope peer's internal_metadata.

    Uses a single-statement nested JSONB merge so concurrent writers updating
    *different* session keys never clobber each other (the merge is computed
    from the row's current committed value under the row lock, never from a
    stale Python-side read):

        internal_metadata || {"backfill_status":
            coalesce(internal_metadata->'backfill_status', '{}') || {<session>: <entry>}}

    Does not commit; the caller owns the transaction. Callers should
    invalidate the peer cache after committing (``peer_cache_key``).
    """
    entry: dict[str, Any] = {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if docs_copied is not None:
        entry["docs_copied"] = docs_copied

    # NOTE: the JSONB operand must be a Python dict, not a json.dumps() string.
    # cast(<already-serialized-str>, JSONB) double-encodes: psycopg's JSONB
    # bind adapter serializes the *string* again, producing a JSONB string
    # scalar instead of an object. `||` between two non-array jsonb scalars
    # doesn't merge keys — it silently wraps both sides into a 2-element
    # array, corrupting backfill_status into a list and later crashing
    # clear_scope_backfill_status's `#-` path delete (which then sees an
    # array where it expects an object).
    merged_status = func.coalesce(
        models.Peer.internal_metadata.op("->")(BACKFILL_STATUS_KEY),
        cast({}, JSONB),
    ).op("||")(cast({session_name: entry}, JSONB))

    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == scope_peer)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                func.jsonb_build_object(BACKFILL_STATUS_KEY, merged_status)
            )
        )
    )
    result = py_cast(CursorResult[Any], await db.execute(stmt))
    if result.rowcount == 0:
        raise ResourceNotFoundException(
            f"Scope peer {scope_peer} not found in workspace {workspace_name}"
        )


async def clear_scope_backfill_status(
    db: AsyncSession,
    workspace_name: str,
    scope_peer: str,
    session_name: str,
) -> None:
    """
    Drop one session's entry from the scope peer's backfill status.

    Called by the removal reconciliation job: once a session leaves a scope its
    backfill status is moot, and clearing it keeps add→remove→re-add cycles
    honest (the re-add starts from a fresh ``pending``). Single-statement
    ``#-`` delete, safe against concurrent per-key merges. Does not commit.
    Missing peers are a no-op (removal is idempotent).
    """
    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == scope_peer)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("#-")(
                cast(array([BACKFILL_STATUS_KEY, session_name]), ARRAY(Text))
            )
        )
    )
    await db.execute(stmt)


async def invalidate_scope_peer_cache(workspace_name: str, scope_peer: str) -> None:
    """Invalidate the cached peer row after a status write commits."""
    await safe_cache_delete(peer_cache_key(workspace_name, scope_peer))


async def get_scope_backfill_status(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
) -> dict[str, Any]:
    """
    Read the per-session backfill job status map for a scope.

    Returns:
        ``{<session_name>: {state, updated_at[, docs_copied]}}`` (empty when no
        backfill has ever been enqueued for the scope)

    Raises:
        ResourceNotFoundException: If the scope does not exist
    """
    peer = await get_scope_or_raise(db, workspace_name, scope_name)
    status = peer.internal_metadata.get(BACKFILL_STATUS_KEY, {})
    if not isinstance(status, dict):
        return {}
    return py_cast(dict[str, Any], status)
