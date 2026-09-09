"""CRUD helpers for peer records and peer-scoped session queries."""

import re
from collections.abc import Collection, Iterable
from logging import getLogger
from typing import Any, Literal

from cashews import NOT_NONE
from sqlalchemy import ColumnElement, Select, and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src import models, schemas
from src.cache.client import (
    cache,
    cache_key_namespace,
    cache_prefix_namespace,
    safe_cache_delete,
)
from src.config import settings
from src.crud.workspace import get_or_create_workspace
from src.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)
from src.models import Peer
from src.schemas.api import RESOURCE_NAME_PATTERN
from src.utils import scopes as scopes_util
from src.utils.filter import apply_filter
from src.utils.types import GetOrCreateResult

logger = getLogger(__name__)

# Matches the peers.name CHECK constraint and PeerCreate's max_length.
PEER_NAME_MAX_LENGTH = 512

PEER_CACHE_KEY_TEMPLATE = "v2:workspace:{workspace_name}:peer:{peer_name}"
PEER_LOCK_PREFIX = f"{cache_prefix_namespace()}:lock:v2"


def peer_cache_key(workspace_name: str, peer_name: str) -> str:
    """Generate cache key for peer."""
    return (
        cache_key_namespace()
        + ":"
        + PEER_CACHE_KEY_TEMPLATE.format(
            workspace_name=workspace_name,
            peer_name=peer_name,
        )
    )


def _reject_impossible_peer_names(names: Collection[str]) -> None:
    """Reject names that cannot correspond to any stored row, before querying.

    ``PeerSpec`` accepts anything so existing names can be looked up, and the
    full new-name rules run later on the insert path — but a couple of values
    cannot be a legacy row *by construction*, and sending them to Postgres first
    fails before that 422 can happen:

    - NUL bytes: Postgres text cannot hold them, so psycopg raises DataError
      during the lookup itself, surfacing as a 500.
    - Over-length names: the ``peers.name`` CHECK caps them at
      ``PEER_NAME_MAX_LENGTH``, so no stored row can exceed it.

    Takes a ``Collection`` rather than an ``Iterable`` on purpose: it inspects the
    input twice, so a generator would be half-consumed and the second check would
    silently see nothing.

    Raises:
        ValidationException: On a NUL byte or an over-length name.
    """
    if any("\x00" in name for name in names):
        raise ValidationException("Peer name(s) must not contain NUL (0x00) bytes")
    too_long = sorted({n for n in names if len(n) > PEER_NAME_MAX_LENGTH})
    if too_long:
        raise ValidationException(
            f"Peer name(s) {too_long} must be at most "
            + f"{PEER_NAME_MAX_LENGTH} characters"
        )


def _validate_new_peer_names(names: list[str]) -> None:
    """Validate peer names that are about to be created.

    Mirrors ``PeerCreate``'s contract for peers arriving through crud rather than
    the peers route. The reserved prefix is reported separately because it is
    also outside ``RESOURCE_NAME_PATTERN``, so the charset check would otherwise
    mask the real problem.

    Raises:
        ValidationException: On a reserved-prefix or non-conforming name.
    """
    scopes_util.validate_no_scope_peer_names(
        names, action="Use the scopes routes to create scopes."
    )
    # Length and NUL bytes are already refused before the lookup by
    # _reject_impossible_peer_names; RESOURCE_NAME_PATTERN's `+` rejects empty.
    offenders = sorted({n for n in names if not re.fullmatch(RESOURCE_NAME_PATTERN, n)})
    if offenders:
        raise ValidationException(
            f"Peer name(s) {offenders} must match pattern {RESOURCE_NAME_PATTERN}"
        )


def scope_peer_clause() -> ColumnElement[bool]:
    """SQL form of ``is_scope_peer()``: reserved name prefix AND the internal kind flag.

    Lives here rather than in ``crud/scope.py`` because that module already imports
    from this one, and ``get_peers`` below needs the clause — the other direction
    would be a cycle.

    ``autoescape=True`` is future-proofing: '.' is not a LIKE wildcard, but '_' is,
    so under a ``scope__``-style prefix an unescaped ``startswith`` would also match
    ``scopeXY...``. Both columns are NOT NULL with defaults, so the negation
    ``~scope_peer_clause()`` has no NULL-semantics trap.
    """
    return and_(
        models.Peer.name.startswith(scopes_util.SCOPE_PEER_PREFIX, autoescape=True),
        models.Peer.internal_metadata.contains({"kind": scopes_util.SCOPE_KIND}),
    )


def _reserved_name_candidates(names: Iterable[str]) -> list[str]:
    """Materialize ``names`` once and return the reserved-prefix ones, sorted.

    Materializing up front matters: callers pass generators (the message-author
    path does), and validating impossible names iterates the input separately from
    the prefix filter — a generator would be silently half-consumed.

    Impossible values are refused here, before any SQL, because a reserved-prefix
    name containing a NUL byte would otherwise reach the text comparison below and
    raise ``psycopg.DataError`` inside the query — a 500 instead of the 422 the
    caller should get.

    Raises:
        ValidationException: On a NUL byte or an over-length name.
    """
    materialized = tuple(names)
    _reject_impossible_peer_names(materialized)
    return sorted({n for n in materialized if scopes_util.is_scope_peer_name(n)})


async def reject_scope_observed(
    db: AsyncSession,
    workspace_name: str,
    names: Iterable[str],
    *,
    action: str,
) -> None:
    """Reject any name that is — or could later become — an observed scope.

    Stricter than ``reject_scope_peers`` in exactly one case: a **missing**
    reserved name is refused. Use this for the *observed* position, where nothing
    creates the peer and so nothing else would ever catch it. Without it a caller
    can pre-seed state about ``scope.future`` while that peer does not exist, then
    create the scope and have the state retroactively describe it.

    Three-way on the reserved namespace:

    ==============================  ======
    State                           Result
    ==============================  ======
    Existing flagged scope          reject
    Missing reserved name           reject
    Existing unflagged squatter     allow
    ==============================  ======

    Non-reserved names are left entirely to the caller's own existence semantics.

    Raises:
        ValidationException: On a real scope or a missing reserved name.
    """
    candidates = _reserved_name_candidates(names)
    if not candidates:
        return

    rows = (
        await db.execute(
            select(models.Peer.name, scope_peer_clause())
            .where(models.Peer.workspace_name == workspace_name)
            .where(models.Peer.name.in_(candidates))
        )
    ).all()
    flagged = {name for name, is_scope in rows if is_scope}
    existing = {name for name, _ in rows}

    scopes = sorted(flagged)
    if scopes:
        raise ValidationException(f"Peer name(s) {scopes} are scopes. {action}")

    missing = sorted(set(candidates) - existing)
    if missing:
        raise ValidationException(
            f"Peer name(s) {missing} are in the reserved scope namespace and do"
            + f" not exist, so they may become scopes later. {action}"
        )


async def scope_peer_names(
    db: AsyncSession,
    workspace_name: str,
    names: Iterable[str],
) -> set[str]:
    """Return the subset of ``names`` that are really scope peers (name AND flag).

    Unlike a pure name check, a legacy peer that merely occupies the reserved
    namespace (names were length-only validated before migration
    ``d429de0e5338``, so ``scope.production`` is a possible user name) is not
    reported, so it keeps its ordinary semantics instead of being locked out of
    its own data. A *missing* reserved name is likewise not reported.

    Costs nothing on the common path: with no reserved-prefix name in ``names``
    there is no query at all.

    Raises:
        ValidationException: On a NUL byte or an over-length name.
    """
    candidates = _reserved_name_candidates(names)
    if not candidates:
        return set()

    result = await db.execute(
        select(models.Peer.name)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(candidates))
        .where(scope_peer_clause())
    )
    return {row[0] for row in result.all()}


async def reject_scope_peers(
    db: AsyncSession,
    workspace_name: str,
    names: Iterable[str],
    *,
    action: str,
) -> None:
    """Reject peers that really are scopes, keyed off name AND flag.

    A *missing* reserved name passes here — the create paths this guards
    (`get_or_create_peers`) refuse it themselves. Positions where nothing creates
    the peer need ``reject_scope_observed`` instead. See ``scope_peer_names`` for
    the name-vs-flag semantics.

    Raises:
        ValidationException: If any name resolves to a real scope peer.
    """
    offenders = sorted(await scope_peer_names(db, workspace_name, names))
    if offenders:
        raise ValidationException(f"Peer name(s) {offenders} are scopes. {action}")


async def get_or_create_peers(
    db: AsyncSession,
    workspace_name: str,
    peers: list[schemas.PeerSpec],
    *,
    _retry: bool = False,
    _pending_invalidation: list[str] | None = None,
) -> GetOrCreateResult[list[models.Peer]]:
    """
    Get an existing list of peers or create new peers if they don't exist.
    Updates existing peers with metadata and configuration if provided.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peers: List of peer creation schemas
        _retry: Whether to retry the operation
        _pending_invalidation: Names of peers already mutated by a prior attempt,
            whose cache keys must still be purged. See the retry branch below.

    Returns:
        GetOrCreateResult containing the list of peers and whether any were created

    Raises:
        ConflictException: If we fail to get or create the peers
        ValidationException: On an impossible name (NUL byte, over-length), or a
            reserved-prefix or non-conforming name on the create path
    """

    await get_or_create_workspace(db, schemas.WorkspaceCreate(name=workspace_name))
    peer_names = [p.name for p in peers]
    # Before the lookup: these values cannot match a stored row and would fail
    # inside the query itself rather than as a clean 422.
    _reject_impossible_peer_names(peer_names)
    stmt = (
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name.in_(peer_names))
    )
    result = await db.execute(stmt)
    existing_peers: list[Peer] = list(result.scalars().all())

    # Create a mapping of peer names to peer schemas for easy lookup
    peer_schema_map = {p.name: p for p in peers}

    # Track which peers actually changed
    changed_peers: list[Peer] = []

    # Update existing peers with metadata and configuration if provided
    for existing_peer in existing_peers:
        peer_schema = peer_schema_map[existing_peer.name]
        changed = False

        # Update with metadata if provided AND different
        if (
            peer_schema.metadata is not None
            and existing_peer.h_metadata != peer_schema.metadata
        ):
            existing_peer.h_metadata = peer_schema.metadata
            changed = True

        # Update with configuration if provided AND different
        if (
            peer_schema.configuration is not None
            and existing_peer.configuration != peer_schema.configuration
        ):
            existing_peer.configuration = peer_schema.configuration
            changed = True

        if changed:
            changed_peers.append(existing_peer)

    # Find which peers need to be created
    existing_names = {p.name for p in existing_peers}
    peers_to_create = [p for p in peers if p.name not in existing_names]

    # Names are validated on the *create* path only. `PeerSpec` deliberately
    # carries no charset pattern so already-existing names (legacy dotted names,
    # scope peers) can be looked up without a spurious 422 — but a name we are
    # about to INSERT is a new peer, and new peers must obey the public contract.
    # Without this, request-controlled names reach here unvalidated via message
    # authors, session peer maps, and the chat observer path, letting a caller
    # mint `scope.x` squatters (permanently 409-blocking that scope) or peers
    # that violate RESOURCE_NAME_PATTERN outright.
    if peers_to_create:
        _validate_new_peer_names([p.name for p in peers_to_create])

    # Create new peers
    new_peers = [
        models.Peer(
            workspace_name=workspace_name,
            name=p.name,
            h_metadata=p.metadata or {},
            configuration=p.configuration or {},
        )
        for p in peers_to_create
    ]
    try:
        async with db.begin_nested():
            db.add_all(new_peers)
    except IntegrityError:
        if _retry:
            raise ConflictException(
                f"Unable to create or get peers: {peer_names}"
            ) from None
        # `begin_nested()` autoflushes the mutations above *before* opening the
        # savepoint, so they are already committed-in-transaction and the rollback
        # doesn't undo them — nor does it expire the now-clean ORM state. The retry
        # would therefore compare already-updated values, find no change, and skip
        # the purge. Carry the names forward so the invalidation can't be lost.
        return await get_or_create_peers(
            db,
            workspace_name,
            peers,
            _retry=True,
            _pending_invalidation=(_pending_invalidation or [])
            + [p.name for p in changed_peers],
        )

    # Capture peer names eagerly so the closure holds plain strings, not ORM objects
    _cache_keys_to_invalidate = [
        peer_cache_key(workspace_name, name)
        for name in dict.fromkeys(
            (_pending_invalidation or []) + [p.name for p in changed_peers + new_peers]
        )
    ]

    async def _invalidate_peer_cache():
        for cache_key in _cache_keys_to_invalidate:
            await safe_cache_delete(cache_key)

    # Return combined list of existing and new peers
    # created=True if any new peers were created
    return GetOrCreateResult(
        existing_peers + new_peers,
        created=len(new_peers) > 0,
        on_commit=_invalidate_peer_cache if _cache_keys_to_invalidate else None,
    )


@cache(
    key=PEER_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_TTL_SECONDS}s",
    prefix=cache_prefix_namespace(),
    condition=NOT_NONE,
)
@cache.locked(
    key=PEER_CACHE_KEY_TEMPLATE,
    ttl=f"{settings.CACHE.DEFAULT_LOCK_TTL_SECONDS}s",
    prefix=PEER_LOCK_PREFIX,
    check_interval=settings.CACHE.LOCK_WAIT_CHECK_INTERVAL_SECONDS,
)
async def _fetch_peer(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
) -> dict[str, Any] | None:
    """Fetch a peer from the database and return as a plain dict for safe caching."""
    obj = await db.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == peer_name)
    )
    if obj is None:
        return None
    return {
        "id": obj.id,
        "name": obj.name,
        "workspace_name": obj.workspace_name,
        "h_metadata": obj.h_metadata,
        "internal_metadata": obj.internal_metadata,
        "configuration": obj.configuration,
        "created_at": obj.created_at,
    }


async def get_peer(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
) -> models.Peer:
    """
    Get an existing peer.

    Takes a plain name, not a create schema: this is a pure read, and validating
    an already-existing name against ``PeerCreate``'s charset pattern turns a
    lookup into a raw pydantic ValidationError (an HTTP 500) for legacy dotted
    names and every ``scope.``-prefixed peer.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer

    Returns:
        The peer if found

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    data = await _fetch_peer(db, workspace_name, peer_name)
    if data is None:
        raise ResourceNotFoundException(
            f"Peer {peer_name} not found in workspace {workspace_name}"
        )

    # Reconstruct ORM object from cached dict and merge into session
    obj = models.Peer(**data)
    make_transient_to_detached(obj)
    existing_peer = await db.merge(obj, load=False)

    return existing_peer


async def get_peers(
    workspace_name: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    kind: Literal["scope", "all"] | None = None,
) -> Select[tuple[models.Peer]]:
    """Build a filtered peer list query ordered by creation time.

    Args:
        workspace_name: Name of the workspace
        filters: Filter peers by metadata
        reverse: Whether to reverse the default creation order
        kind: Which kinds of peers to include. None (default) excludes scope
            peers (see ``scope_peer_clause``: reserved name prefix AND the
            ``{"kind": "scope"}`` internal_metadata flag), "scope" returns only
            scope peers, and "all" returns everything.
    """
    stmt = select(models.Peer).where(models.Peer.workspace_name == workspace_name)

    if kind is None:
        stmt = stmt.where(~scope_peer_clause())
    elif kind == "scope":
        stmt = stmt.where(scope_peer_clause())

    stmt = apply_filter(stmt, models.Peer, filters)

    if reverse:
        return stmt.order_by(models.Peer.created_at.desc(), models.Peer.id.desc())
    return stmt.order_by(models.Peer.created_at.asc(), models.Peer.id.asc())


async def update_peer(
    db: AsyncSession, workspace_name: str, peer_name: str, peer: schemas.PeerUpdate
) -> models.Peer:
    """
    Get or create a peer, then apply metadata and configuration updates.

    If the peer does not exist, the workspace and peer are created first.
    Provided metadata and configuration replace the existing values when
    present.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        peer: Peer update schema

    Returns:
        The updated peer

    Raises:
        ConflictException: If concurrent creation prevents fetching or creating
            the peer
    """
    peers_result = await get_or_create_peers(
        db, workspace_name, [schemas.PeerSpec(name=peer_name)]
    )
    honcho_peer = peers_result.resource[0]

    # Refuse a real scope on the row just resolved, not on the name beforehand:
    # this route replaces `configuration` wholesale, and a name-level check leaves
    # a window in which a concurrently-created scope is resolved as existing (so
    # create-path validation never fires) and then overwritten. An existing
    # *unflagged* peer in the reserved namespace is an ordinary peer and passes.
    if scopes_util.is_scope_peer(honcho_peer.name, honcho_peer.internal_metadata):
        raise ValidationException(
            f"Peer '{peer_name}' is a scope."
            + " Use the scopes routes to manage scopes."
        )

    needs_update = False

    if peer.metadata is not None and honcho_peer.h_metadata != peer.metadata:
        honcho_peer.h_metadata = peer.metadata
        needs_update = True

    if (
        peer.configuration is not None
        and honcho_peer.configuration != peer.configuration
    ):
        honcho_peer.configuration = peer.configuration
        needs_update = True

    # Early exit if unchanged
    if not needs_update:
        await db.commit()
        await peers_result.post_commit()
        logger.debug(
            "Peer %s unchanged in workspace %s, skipping update",
            peer_name,
            workspace_name,
        )
        return honcho_peer

    await db.commit()
    await peers_result.post_commit()

    cache_key = peer_cache_key(workspace_name, honcho_peer.name)
    await safe_cache_delete(cache_key)

    logger.debug("Peer %s updated successfully", peer_name)
    return honcho_peer


async def get_sessions_for_peer(
    workspace_name: str,
    peer_name: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
) -> Select[tuple[models.Session]]:
    """
    Get all sessions for a peer through the session_peers relationship.

    Args:
        workspace_name: Name of the workspace
        peer_name: Name of the peer
        filters: Filter sessions by metadata
        reverse: Whether to reverse the default creation order

    Returns:
        SQLAlchemy Select statement
    """
    stmt = (
        select(models.Session)
        .join(
            models.SessionPeer,
            (models.Session.name == models.SessionPeer.session_name)
            & (models.Session.workspace_name == models.SessionPeer.workspace_name),
        )
        .where(models.SessionPeer.peer_name == peer_name)
        .where(models.Session.workspace_name == workspace_name)
    )

    stmt = apply_filter(stmt, models.Session, filters)

    if reverse:
        stmt = stmt.order_by(models.Session.created_at.desc(), models.Session.id.desc())
    else:
        stmt = stmt.order_by(models.Session.created_at.asc(), models.Session.id.asc())

    return stmt
