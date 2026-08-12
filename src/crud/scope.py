"""CRUD helpers for scopes.

A scope is a named grouping of sessions, implemented as a peer named
``scope.<name>`` carrying ``{"kind": "scope"}`` in ``internal_metadata`` (the
authoritative, user-unwritable flag) and ``{"observe_me": false}`` in
``configuration``, that observes its member sessions (``observe_others=true``)
and never speaks.
See ``src/utils/scopes.py`` for the namespace helpers.

Membership only affects messages ingested *after* a session is added to a
scope. Conclusions already derived are neither backfilled on add nor
reconciled on removal.
"""

from logging import getLogger

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models, schemas
from src.cache.client import safe_cache_delete
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.scopes import (
    SCOPE_KIND,
    is_scope_peer,
    scope_peer_name,
)
from src.utils.types import GetOrCreateResult

from .peer import peer_cache_key, scope_peer_clause
from .workspace import get_or_create_workspace

logger = getLogger(__name__)

# Internal metadata stamped on every scope peer at creation. `kind` is the
# authoritative scope flag and lives here — NOT in `configuration` — because
# `configuration` is user-writable (`PeerCreate`/`PeerUpdate` accept a free-form
# dict, and `update_peer` replaces it wholesale), so a user could forge or clear
# the flag. `internal_metadata` appears in no API schema at all.
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
    observer peer would carry. No backfill happens here: membership only affects
    messages ingested after this call, and conclusions already derived are left
    as they are.

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


async def remove_session_from_scope(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
    session_name: str,
) -> None:
    """
    Remove a session from a scope by ending the scope peer's membership.

    Ends the membership the same way the generic remove-peer path does (sets
    ``left_at``). Conclusions derived while the session was a member are left in
    place — nothing reconciles them.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name
        session_name: Name of the session to remove

    Raises:
        ResourceNotFoundException: If the scope or session does not exist
    """
    # Lazy import for the same circular-import reason as add_sessions_to_scope.
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
