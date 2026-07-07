"""CRUD helpers for scopes.

A scope is a named grouping of sessions, implemented as a peer named
``scope__<name>`` with configuration ``{"kind": "scope", "observe_me": false}``
that observes its member sessions (``observe_others=true``) and never speaks.
See ``src/utils/scopes.py`` for the namespace helpers.

Membership only affects messages ingested *after* a session is added to a
scope; backfill of pre-existing documents and reconciliation on removal land
in a follow-up (DEV-1999).
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
    is_scope_peer_configuration,
    scope_peer_name,
)
from src.utils.types import GetOrCreateResult

from .peer import peer_cache_key
from .workspace import get_or_create_workspace

logger = getLogger(__name__)

# Peer-level configuration stamped on every scope peer at creation. `kind` is
# the authoritative scope flag; `observe_me: false` ensures no representation
# is ever formed *of* a scope peer.
SCOPE_PEER_CONFIGURATION: dict[str, str | bool] = {
    "kind": SCOPE_KIND,
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
) -> GetOrCreateResult[list[models.Peer]]:
    """
    Get existing scopes or create new ones if they don't exist.

    Existing scope peers have their metadata updated when provided. A
    pre-existing peer that occupies a scope's reserved name *without* the
    authoritative ``kind`` flag (a legacy collision) is never adopted.

    Note: does not commit; the caller owns the transaction (mirror of
    ``get_or_create_peers``). Run ``result.post_commit()`` after committing.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scopes: List of scope creation schemas (unprefixed names)

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
        if not is_scope_peer_configuration(existing_peer.configuration):
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
        return await get_or_create_scopes(db, workspace_name, scopes, _retry=True)

    _cache_keys_to_invalidate = [
        peer_cache_key(workspace_name, p.name) for p in changed_peers + new_peers
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
    """Build a scope list query (peers with the scope kind flag) ordered by creation time."""
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.configuration.contains({"kind": SCOPE_KIND}))
    )
    if reverse:
        return stmt.order_by(models.Peer.created_at.desc(), models.Peer.id.desc())
    return stmt.order_by(models.Peer.created_at.asc(), models.Peer.id.asc())


async def get_scope(
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
    if peer is None or not is_scope_peer_configuration(peer.configuration):
        raise ResourceNotFoundException(
            f"Scope {scope_name} not found in workspace {workspace_name}"
        )
    return peer


async def get_scope_session_names(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
) -> list[str]:
    """
    List the IDs of the active sessions that are members of a scope.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name

    Returns:
        Names of the scope's member sessions, oldest membership first

    Raises:
        ResourceNotFoundException: If the scope does not exist
    """
    await get_scope(db, workspace_name, scope_name)

    stmt = (
        select(models.SessionPeer.session_name)
        .join(
            models.Session,
            (models.Session.name == models.SessionPeer.session_name)
            & (models.Session.workspace_name == models.SessionPeer.workspace_name),
        )
        .where(models.SessionPeer.workspace_name == workspace_name)
        .where(models.SessionPeer.peer_name == scope_peer_name(scope_name))
        .where(models.SessionPeer.left_at.is_(None))
        .where(models.Session.is_active == True)  # noqa: E712
        .order_by(models.SessionPeer.joined_at.asc())
    )
    result = await db.execute(stmt)
    return [row[0] for row in result.all()]


async def add_sessions_to_scope(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
    session_names: list[str],
) -> list[str]:
    """
    Add sessions to a scope by creating observer memberships for its peer.

    Each membership is a ``session_peers`` row for the scope peer with
    ``observe_others=true, observe_me=false`` — exactly what a hand-built
    observer peer would carry. Membership only affects messages ingested
    after this call (backfill is DEV-1999).

    Args:
        db: Database session
        workspace_name: Name of the workspace
        scope_name: Unprefixed scope name
        session_names: Names of existing sessions to add

    Returns:
        Names of all the scope's member sessions after the addition

    Raises:
        ResourceNotFoundException: If the scope or any named session does not
            exist
        ObserverException: If a membership would exceed a session's observer
            limit
    """
    # Imported lazily: crud.session imports this module for the session-create
    # `scopes` path, so a module-level import would be circular.
    from .session import upsert_session_peers

    await get_scope(db, workspace_name, scope_name)

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

    return await get_scope_session_names(db, workspace_name, scope_name)


async def remove_session_from_scope(
    db: AsyncSession,
    workspace_name: str,
    scope_name: str,
    session_name: str,
) -> None:
    """
    Remove a session from a scope by ending the scope peer's membership.

    Ends the membership the same way the generic remove-peer path does (sets
    ``left_at``). Reconciliation of documents derived while the session was a
    member lands in a follow-up (DEV-1999).

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

    await get_scope(db, workspace_name, scope_name)

    await remove_peers_from_session(
        db,
        workspace_name=workspace_name,
        session_name=session_name,
        peer_names={scope_peer_name(scope_name)},
    )
