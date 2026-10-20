"""CRUD helpers for peer memberships and the read-time expansion over them.

A membership says one peer belongs to another within a workspace. Rows are keyed
by peer id on both sides; the API speaks peer names, so every helper here
resolves names at its edge. Nothing in this module touches conclusions: a read
that names a target expands the target into its members with
``expand_observed`` and queries each one's own representation.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from sqlalchemy import Row, Select, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from src import models, schemas
from src.config import settings
from src.crud.document import ObservedPair
from src.crud.peer import get_or_create_peers, scope_peer_clause
from src.exceptions import ConflictException, ValidationException
from src.utils.filter import apply_filter
from src.utils.scopes import is_scope_peer

logger = getLogger(__name__)

# Upper bound on how far the cycle check walks up from a proposed parent. The
# graph is kept acyclic, so the walk ends on its own; the bound only keeps one
# pathological chain from making a membership write unbounded.
MAX_ANCESTOR_WALK = 1000


@dataclass(frozen=True)
class AddMemberResult:
    membership: schemas.PeerMembership
    created: bool
    # Emits events and invalidates caches for any peers this created. Call it
    # after the transaction commits.
    post_commit: Callable[[], Awaitable[None]]


def _membership_select() -> tuple[Select[Any], Any, Any]:
    """Membership rows with both peers' names, as ``(membership, parent, member)``."""
    parent = aliased(models.Peer, name="parent_peer")
    member = aliased(models.Peer, name="member_peer")
    stmt = (
        select(models.PeerMembership, parent.name, member.name)
        .join(parent, parent.id == models.PeerMembership.parent_id)
        .join(member, member.id == models.PeerMembership.member_id)
    )
    return stmt, parent, member


def membership_rows_to_schemas(
    rows: Sequence[Row[tuple[models.PeerMembership, str, str]]],
) -> list[schemas.PeerMembership]:
    """Pagination transformer for ``get_members`` and ``get_parents``."""
    return [
        _to_schema(membership, parent_name, member_name)
        for membership, parent_name, member_name in rows
    ]


def _lock_workspace_memberships(workspace_name: str) -> Any:
    """Serialize membership writes in one workspace for the transaction.

    Two concurrent inserts can each pass the cycle check and together close a
    cycle (a in b, b in a). Holding this lock across check and write makes them
    take turns.
    """
    return text(
        "SELECT pg_advisory_xact_lock(hashtext('peer_memberships'), "
        + "hashtext(:workspace_name))"
    ).bindparams(workspace_name=workspace_name)


async def _ancestor_path(
    db: AsyncSession, workspace_name: str, *, start_id: str, target_id: str
) -> list[str] | None:
    """Return a parent chain from ``start_id`` up to ``target_id``, if one exists.

    The chain is a list of peer ids beginning at ``start_id`` and ending at
    ``target_id``. Walks up the membership graph (member -> parent) one level
    per query, visiting each peer once, so the cost is linear in the peers and
    memberships above ``start_id`` however many paths lead through them.

    Raises:
        ValidationException: If the walk exceeds ``MAX_ANCESTOR_WALK`` levels.
    """
    reached_from: dict[str, str | None] = {start_id: None}
    frontier = [start_id]
    for _ in range(MAX_ANCESTOR_WALK):
        if not frontier:
            return None
        rows = await db.execute(
            select(models.PeerMembership.member_id, models.PeerMembership.parent_id)
            .where(models.PeerMembership.workspace_name == workspace_name)
            .where(models.PeerMembership.member_id.in_(frontier))
        )
        next_frontier: list[str] = []
        for member_id, parent_id in rows.all():
            if parent_id in reached_from:
                continue
            reached_from[parent_id] = member_id
            if parent_id == target_id:
                path = [parent_id]
                while (previous := reached_from[path[-1]]) is not None:
                    path.append(previous)
                return list(reversed(path))
            next_frontier.append(parent_id)
        frontier = next_frontier
    if frontier:
        raise ValidationException(
            f"Membership chain is deeper than {MAX_ANCESTOR_WALK} levels"
        )
    return None


async def add_member(
    db: AsyncSession,
    workspace_name: str,
    *,
    parent_name: str,
    member_name: str,
    metadata: dict[str, Any] | None = None,
) -> AddMemberResult:
    """Add ``member_name`` as a member of ``parent_name``, creating peers as needed.

    Adding an existing membership again is not an error; it replaces the
    membership's metadata when ``metadata`` is given and reports
    ``created=False``.

    Raises:
        ValidationException: If either peer is a scope, the peer names itself,
            or the parent already has ``PEER_MAX_MEMBERS`` members.
        ConflictException: If the membership would close a cycle.
    """
    if parent_name == member_name:
        raise ValidationException("A peer cannot be a member of itself")

    # Sorted so two requests creating the same pair of new peers in opposite
    # roles insert them in the same order rather than deadlocking.
    peers_result = await get_or_create_peers(
        db,
        workspace_name=workspace_name,
        peers=[
            schemas.PeerSpec(name=name) for name in sorted((parent_name, member_name))
        ],
    )
    by_name = {peer.name: peer for peer in peers_result.resource}
    parent, member = by_name[parent_name], by_name[member_name]
    scopes = sorted(
        peer.name
        for peer in (parent, member)
        if is_scope_peer(peer.name, peer.internal_metadata)
    )
    if scopes:
        raise ValidationException(
            f"Peer name(s) {scopes} are scopes. A scope cannot be a member or a "
            + "parent: use the scope's sessions instead."
        )

    await db.execute(_lock_workspace_memberships(workspace_name))

    existing = await db.scalar(
        select(models.PeerMembership)
        .where(models.PeerMembership.workspace_name == workspace_name)
        .where(models.PeerMembership.parent_id == parent.id)
        .where(models.PeerMembership.member_id == member.id)
    )
    if existing is not None:
        if metadata is not None:
            existing.h_metadata = metadata
        await db.flush()
        return AddMemberResult(
            membership=_to_schema(existing, parent.name, member.name),
            created=False,
            post_commit=peers_result.post_commit,
        )

    member_count = await db.scalar(
        select(func.count())
        .select_from(models.PeerMembership)
        .where(models.PeerMembership.workspace_name == workspace_name)
        .where(models.PeerMembership.parent_id == parent.id)
    )
    if (member_count or 0) >= settings.PEER.MAX_MEMBERS:
        raise ValidationException(
            f"Peer {parent_name} already has the maximum of "
            + f"{settings.PEER.MAX_MEMBERS} members"
        )

    # parent -> member closes a cycle exactly when member is already an
    # ancestor of parent.
    path = await _ancestor_path(
        db, workspace_name, start_id=parent.id, target_id=member.id
    )
    if path is not None:
        names = await _names_for_ids(db, workspace_name, path)
        chain = " -> ".join(names[peer_id] for peer_id in reversed(path))
        raise ConflictException(
            f"Adding {member_name} to {parent_name} would create a cycle: "
            + f"{chain} -> {member_name}"
        )

    membership = models.PeerMembership(
        workspace_name=workspace_name,
        parent_id=parent.id,
        member_id=member.id,
        h_metadata=metadata or {},
    )
    db.add(membership)
    await db.flush()
    await db.refresh(membership)
    return AddMemberResult(
        membership=_to_schema(membership, parent.name, member.name),
        created=True,
        post_commit=peers_result.post_commit,
    )


async def _names_for_ids(
    db: AsyncSession, workspace_name: str, peer_ids: list[str]
) -> dict[str, str]:
    rows = await db.execute(
        select(models.Peer.id, models.Peer.name)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.id.in_(peer_ids))
    )
    return {peer_id: name for peer_id, name in rows.all()}


def _to_schema(
    membership: models.PeerMembership, parent_name: str, member_name: str
) -> schemas.PeerMembership:
    return schemas.PeerMembership(
        parent_id=parent_name,
        peer_id=member_name,
        workspace_id=membership.workspace_name,
        metadata=membership.h_metadata,
        created_at=membership.created_at,
    )


async def remove_member(
    db: AsyncSession,
    workspace_name: str,
    *,
    parent_name: str,
    member_name: str,
) -> bool:
    """Remove a membership. Returns whether a row was removed; missing is not an error."""
    parent = aliased(models.Peer)
    member = aliased(models.Peer)
    ids = (
        select(models.PeerMembership.id)
        .join(parent, parent.id == models.PeerMembership.parent_id)
        .join(member, member.id == models.PeerMembership.member_id)
        .where(models.PeerMembership.workspace_name == workspace_name)
        .where(parent.name == parent_name)
        .where(member.name == member_name)
    )
    result = await db.execute(
        delete(models.PeerMembership).where(models.PeerMembership.id.in_(ids))
    )
    return bool(getattr(result, "rowcount", 0))


def get_members(
    workspace_name: str,
    parent_name: str,
    filters: dict[str, Any] | None = None,
) -> Select[Any]:
    """Select a peer's members, oldest membership first. ``filters`` apply to the member peer."""
    stmt, parent, member = _membership_select()
    stmt = (
        stmt.where(models.PeerMembership.workspace_name == workspace_name)
        .where(parent.name == parent_name)
        .order_by(models.PeerMembership.created_at, models.PeerMembership.id)
    )
    return apply_filter(stmt, member, filters)


def get_parents(
    workspace_name: str,
    member_name: str,
    filters: dict[str, Any] | None = None,
) -> Select[Any]:
    """Select the peers a peer belongs to, oldest membership first. ``filters`` apply to the parent peer."""
    stmt, parent, member = _membership_select()
    stmt = (
        stmt.where(models.PeerMembership.workspace_name == workspace_name)
        .where(member.name == member_name)
        .order_by(models.PeerMembership.created_at, models.PeerMembership.id)
    )
    return apply_filter(stmt, parent, filters)


def resolve_depth(depth: int | None) -> int:
    """Apply the configured default to a requested depth and enforce the maximum.

    Raises:
        ValidationException: If ``depth`` exceeds ``PEER_AGGREGATION_MAX_DEPTH``.
    """
    if depth is None:
        return settings.PEER.AGGREGATION_DEFAULT_DEPTH
    if depth > settings.PEER.AGGREGATION_MAX_DEPTH:
        raise ValidationException(
            f"depth {depth} exceeds the maximum of "
            + f"{settings.PEER.AGGREGATION_MAX_DEPTH}. Lower the depth, or query "
            + "a member directly."
        )
    return depth


async def expand_observed(
    db: AsyncSession,
    workspace_name: str,
    target_name: str,
    depth: int,
) -> list[str]:
    """Expand a target peer into itself plus its members down to ``depth``.

    Returns peer names with the target first, then members level by level, each
    level in membership order; a peer reachable along two paths appears once.
    Scope peers are never included, whatever the membership table holds. A
    target with no members expands to ``[target_name]`` at every depth.

    Raises:
        ValidationException: If the set would exceed ``PEER_AGGREGATION_MAX_FANOUT``.
    """
    expanded = [target_name]
    if depth <= 0:
        return expanded

    target_id = await db.scalar(
        select(models.Peer.id)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == target_name)
    )
    if target_id is None:
        return expanded

    seen_ids = {target_id}
    frontier = [target_id]
    fanout_cap = settings.PEER.AGGREGATION_MAX_FANOUT
    for _ in range(depth):
        if not frontier:
            break
        rows = (
            await db.execute(
                select(models.Peer.id, models.Peer.name)
                .join(
                    models.PeerMembership,
                    models.PeerMembership.member_id == models.Peer.id,
                )
                .where(models.PeerMembership.workspace_name == workspace_name)
                .where(models.PeerMembership.parent_id.in_(frontier))
                .where(~scope_peer_clause())
                .order_by(models.PeerMembership.created_at, models.PeerMembership.id)
            )
        ).all()
        next_frontier: list[str] = []
        for peer_id, name in rows:
            if peer_id in seen_ids:
                continue
            seen_ids.add(peer_id)
            next_frontier.append(peer_id)
            expanded.append(name)
            if len(expanded) > fanout_cap:
                raise ValidationException(
                    f"Aggregating {target_name} at depth {depth} reaches more than "
                    + f"{fanout_cap} peers. Lower the depth, or query a member "
                    + "directly."
                )
        frontier = next_frontier
    return expanded


@dataclass(frozen=True)
class ObservedSet:
    """What a read about one target covers once it is expanded over members.

    ``peers`` is the target followed by its members, as ``expand_observed``
    orders them. ``pairs`` is the (observer, observed) collection read for each
    one. A read about a peer's own global representation (observer and target
    the same peer) reads each member's own global representation too; a
    directional or scoped read keeps its observer and reads that observer's
    view of each member.
    """

    peers: list[str]
    pairs: list[ObservedPair]

    @property
    def aggregated(self) -> bool:
        return len(self.peers) > 1

    @property
    def members(self) -> list[str]:
        return self.peers[1:]


async def resolve_observed_set(
    db: AsyncSession,
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    depth: int,
) -> ObservedSet:
    """Expand ``observed`` over its members and pair each peer with its observer.

    Raises:
        ValidationException: If the set would exceed ``PEER_AGGREGATION_MAX_FANOUT``.
    """
    peers = await expand_observed(db, workspace_name, observed, depth)
    if observer == observed:
        pairs = [(peer, peer) for peer in peers]
    else:
        pairs = [(observer, peer) for peer in peers]
    return ObservedSet(peers=peers, pairs=pairs)
