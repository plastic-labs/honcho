"""Tenant-aware deletion lifecycle checks.

Nothing sets ``deleted_at`` yet. Read paths filter on the row-wise ``live_*``
clauses so tombstoned rows and their dependents stay hidden once deletion is
enabled. These check direct parent liveness, not derived-artifact validity,
authorization, or generation barriers. They must not be used as an unlocked
preflight check before a protected write.
"""

from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import ColumnElement, and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, aliased
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.selectable import Exists

from src import models
from src.config import settings

PrimitiveKind = Literal["workspace", "peer", "session", "message"]


def _context_tenant_id() -> str:
    return (
        models.scoped_tenant_id()
        if settings.MULTI_TENANT
        else models.instance_tenant_id()
    )


@dataclass(frozen=True, slots=True)
class TenantContext:
    """An already routed, caller-owned session and its explicit tenant identity.

    Defaults to the bound tenant in shared storage, failing closed if none is
    bound, or the instance tenant in a dedicated database. Cross-tenant service
    readers must supply the target tenant explicitly. This context does not route
    connections, authorize callers, or bind RLS. The caller owns those concerns.
    Do not serialize the session into durable outbox/purge records.
    """

    db: AsyncSession
    tenant_id: str = field(default_factory=_context_tenant_id)

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("Tenant context requires a tenant_id")


@dataclass(frozen=True, slots=True)
class PrimitiveIdentity:
    """Stable tenant-qualified identity, never a resource name.

    workspace_uid is Workspace.id. uid is Workspace/Peer/Session.id, or
    Message.public_id (not the integer Message.id used by queue rows).
    Resolve names under the caller's authorization before creating this value.
    """

    kind: PrimitiveKind
    workspace_uid: str
    uid: str
    tenant_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("Primitive identity requires a tenant_id")
        if self.kind not in ("workspace", "peer", "session", "message"):
            raise ValueError(f"Unsupported primitive kind: {self.kind}")
        if self.kind == "workspace" and self.uid != self.workspace_uid:
            raise ValueError("Workspace identity must match workspace_uid")


def primitive_liveness_clause(identity: PrimitiveIdentity) -> Exists:
    """Build a self-contained EXISTS for one stable identity and its parents.

    All joins include the tenant and the parent's key. Correlation is disabled: this
    clause checks the supplied identity even if the enclosing query joins or
    aliases the same tables. It is not a row-wise filter for an arbitrary list.
    Missing parents fail closed. Session.is_active and membership.left_at are
    separate archive/access states and intentionally do not affect liveness.
    """
    workspace = models.Workspace
    stmt = (
        select(1)
        .select_from(workspace)
        .where(
            workspace.tenant_id == identity.tenant_id,
            workspace.id == identity.workspace_uid,
            workspace.deleted_at.is_(None),
        )
    )

    match identity.kind:
        case "workspace":
            pass
        case "peer":
            stmt = stmt.join(
                models.Peer,
                and_(
                    models.Peer.tenant_id == workspace.tenant_id,
                    models.Peer.workspace_name == workspace.name,
                ),
            ).where(models.Peer.id == identity.uid, models.Peer.deleted_at.is_(None))
        case "session":
            stmt = stmt.join(
                models.Session,
                and_(
                    models.Session.tenant_id == workspace.tenant_id,
                    models.Session.workspace_name == workspace.name,
                ),
            ).where(
                models.Session.id == identity.uid,
                models.Session.deleted_at.is_(None),
            )
        case "message":
            stmt = (
                stmt.join(
                    models.Message,
                    and_(
                        models.Message.tenant_id == workspace.tenant_id,
                        models.Message.workspace_name == workspace.name,
                    ),
                )
                .join(
                    models.Session,
                    and_(
                        models.Session.tenant_id == models.Message.tenant_id,
                        models.Session.workspace_name == models.Message.workspace_name,
                        models.Session.name == models.Message.session_name,
                    ),
                )
                .join(
                    models.Peer,
                    and_(
                        models.Peer.tenant_id == models.Message.tenant_id,
                        models.Peer.workspace_name == models.Message.workspace_name,
                        models.Peer.name == models.Message.peer_name,
                    ),
                )
                .where(
                    models.Message.public_id == identity.uid,
                    models.Message.deleted_at.is_(None),
                    models.Session.deleted_at.is_(None),
                    models.Peer.deleted_at.is_(None),
                )
            )

    return stmt.correlate(None).exists()


async def is_primitive_live(
    context: TenantContext, identity: PrimitiveIdentity
) -> bool:
    """Read direct liveness from the database, bypassing caches/ORM identity maps.

    One SELECT, no flush (including autoflush), commit, rollback, or locks. The
    caller owns the transaction and must explicitly flush any pending changes
    it wants this query to see. A normal session can autobegin a transaction;
    the answer reflects that transaction's database snapshot, not future writes.
    This is NOT the serialized final-write barrier required by producers.
    """
    if context.tenant_id != identity.tenant_id:
        raise ValueError("Primitive identity does not match the context tenant")
    with context.db.no_autoflush:
        return bool(
            await context.db.scalar(select(primitive_liveness_clause(identity)))
        )


def _workspace_live(
    tenant_id: QueryableAttribute[str], workspace_name: QueryableAttribute[str]
) -> ColumnElement[bool]:
    workspace = aliased(models.Workspace)
    return exists().where(
        workspace.tenant_id == tenant_id,
        workspace.name == workspace_name,
        workspace.deleted_at.is_(None),
    )


def _peer_live(
    tenant_id: QueryableAttribute[str],
    workspace_name: QueryableAttribute[str],
    peer_name: QueryableAttribute[str],
) -> ColumnElement[bool]:
    peer = aliased(models.Peer)
    return exists().where(
        peer.tenant_id == tenant_id,
        peer.workspace_name == workspace_name,
        peer.name == peer_name,
        peer.deleted_at.is_(None),
    )


def _session_live(
    tenant_id: QueryableAttribute[str],
    workspace_name: QueryableAttribute[str],
    session_name: QueryableAttribute[str] | QueryableAttribute[str | None],
) -> ColumnElement[bool]:
    session = aliased(models.Session)
    return exists().where(
        session.tenant_id == tenant_id,
        session.workspace_name == workspace_name,
        session.name == session_name,
        session.deleted_at.is_(None),
    )


def live_workspace_row(
    workspace: type[models.Workspace]
    | AliasedClass[models.Workspace] = models.Workspace,
) -> ColumnElement[bool]:
    """Row-wise filter: the workspace is not tombstoned."""
    return workspace.deleted_at.is_(None)


def live_peer_row(
    peer: type[models.Peer] | AliasedClass[models.Peer] = models.Peer,
) -> ColumnElement[bool]:
    """Row-wise filter: neither the peer nor its workspace is tombstoned.

    Parents are checked with correlated semi-joins on their unique keys, and a
    missing parent fails closed.
    """
    return and_(
        peer.deleted_at.is_(None), _workspace_live(peer.tenant_id, peer.workspace_name)
    )


def live_session_row(
    session: type[models.Session] | AliasedClass[models.Session] = models.Session,
) -> ColumnElement[bool]:
    """Row-wise filter: neither the session nor its workspace is tombstoned.

    ``is_active`` is a separate archive state and is not checked here.
    """
    return and_(
        session.deleted_at.is_(None),
        _workspace_live(session.tenant_id, session.workspace_name),
    )


def live_message_row(
    message: type[models.Message] | AliasedClass[models.Message] = models.Message,
) -> ColumnElement[bool]:
    """Row-wise filter: the message, its session, author peer and workspace are live."""
    return and_(
        message.deleted_at.is_(None),
        _workspace_live(message.tenant_id, message.workspace_name),
        _session_live(message.tenant_id, message.workspace_name, message.session_name),
        _peer_live(message.tenant_id, message.workspace_name, message.peer_name),
    )


def live_session_peer_row(
    session_peer: type[models.SessionPeer]
    | AliasedClass[models.SessionPeer] = models.SessionPeer,
) -> ColumnElement[bool]:
    """Row-wise filter: the membership's session, peer and workspace are live.

    ``left_at`` is a separate membership state and is not checked here.
    """
    return and_(
        _workspace_live(session_peer.tenant_id, session_peer.workspace_name),
        _session_live(
            session_peer.tenant_id,
            session_peer.workspace_name,
            session_peer.session_name,
        ),
        _peer_live(
            session_peer.tenant_id, session_peer.workspace_name, session_peer.peer_name
        ),
    )


def live_dream_run_row(
    run: type[models.DreamRun] | AliasedClass[models.DreamRun] = models.DreamRun,
) -> ColumnElement[bool]:
    """Row-wise filter: the run's workspace, observer and observed are live.

    A run describes one representation, so it is hidden once either peer is
    tombstoned, as that representation's conclusions are.
    """
    return and_(
        _workspace_live(run.tenant_id, run.workspace_name),
        _peer_live(run.tenant_id, run.workspace_name, run.observer),
        _peer_live(run.tenant_id, run.workspace_name, run.observed),
    )


def live_dream_run_change_row(
    change: type[models.DreamRunChange]
    | AliasedClass[models.DreamRunChange] = models.DreamRunChange,
) -> ColumnElement[bool]:
    """Row-wise filter: the change's session, if it has one, is live.

    Only removed-conclusion copies carry a session. A copy is hidden once its
    session is tombstoned, as session deletion hard-deletes it. Callers read
    one run's changes or a few conclusion ids, so the check runs per row.
    """
    return or_(
        change.session_name.is_(None),
        _session_live(change.tenant_id, change.workspace_name, change.session_name),
    )


def live_document_row(
    workspace_name: str,
    document: type[models.Document] | AliasedClass[models.Document] = models.Document,
    *,
    tenant_id: str | None = None,
    peer_pair: tuple[str, str] | None = None,
) -> ColumnElement[bool]:
    """Row-wise filter: the conclusion, its workspace, observer and observed are live.

    A conclusion with a ``session_name`` is also hidden once that session is
    tombstoned; one without a session (manual or cross-session) depends only on
    its peers. This hides conclusions under erased parents; it does not decide
    which surviving conclusions were derived from erased messages. Also
    restricts rows to ``workspace_name`` and the supplied tenant, defaulting to
    the caller's bound or instance tenant. Service readers must supply the target
    tenant explicitly when no tenant is bound.

    Conclusion reads scan whole collections, so this does not use the
    correlated semi-joins above. ``session_name IS NULL OR EXISTS (...)``
    cannot become a semi-join, and Postgres instead hashes every live session
    in the database on each read. Here each parent is checked against the
    tenant/workspace's *tombstoned* rows with uncorrelated ``NOT IN`` subqueries, each
    hashed once per query and empty until deletion is enabled. Without a
    partial index on tombstoned peers and sessions, building those hashes
    still reads every peer and session in the workspace. Do not rewrite
    them as ``NOT EXISTS``: the pinned workspace makes Postgres estimate the
    anti-join at about one row, which loses index-ordered plans. The composite
    foreign keys guarantee every parent exists, so this does not need the
    semi-joins' fail-closed check for a missing parent; the workspace is still
    checked positively.

    ``peer_pair`` is for pair-pinned vector reads only: it adds matching outer
    observer/observed equalities and replaces both peer sets with constant,
    positive EXISTS checks. This preserves index-ordered plans without a
    separate probe or snapshot; a missing peer fails closed. The default still
    checks every row's own peers, including when ``document`` is an alias.
    """
    target_tenant = tenant_id if tenant_id is not None else models.scoped_tenant_id()
    if not target_tenant:
        raise ValueError("Conclusion liveness requires a tenant_id")
    workspace = aliased(models.Workspace)
    peer = aliased(models.Peer)
    session = aliased(models.Session)
    peer_liveness: tuple[ColumnElement[bool], ...]
    if peer_pair is None:
        tombstoned_peers = (
            select(peer.name)
            .where(
                peer.tenant_id == target_tenant,
                peer.workspace_name == workspace_name,
                peer.deleted_at.is_not(None),
            )
            .correlate(None)
        )
        peer_liveness = (
            document.observer.not_in(tombstoned_peers),
            document.observed.not_in(tombstoned_peers),
        )
    else:
        observer, observed = peer_pair
        peer_liveness = (
            document.observer == observer,
            document.observed == observed,
            *(
                select(peer.name)
                .where(
                    peer.tenant_id == target_tenant,
                    peer.workspace_name == workspace_name,
                    peer.name == name,
                    peer.deleted_at.is_(None),
                )
                .correlate(None)
                .exists()
                for name in peer_pair
            ),
        )
    tombstoned_sessions = (
        select(session.name)
        .where(
            session.tenant_id == target_tenant,
            session.workspace_name == workspace_name,
            session.deleted_at.is_not(None),
        )
        .correlate(None)
    )
    return and_(
        document.tenant_id == target_tenant,
        document.workspace_name == workspace_name,
        document.deleted_at.is_(None),
        select(workspace.name)
        .where(
            workspace.tenant_id == target_tenant,
            workspace.name == workspace_name,
            workspace.deleted_at.is_(None),
        )
        .correlate(None)
        .exists(),
        *peer_liveness,
        or_(
            document.session_name.is_(None),
            document.session_name.not_in(tombstoned_sessions),
        ),
    )
