"""Deletion lifecycle checks for today's dedicated schema.

Nothing sets ``deleted_at`` yet. Read paths filter on the row-wise ``live_*``
clauses so tombstoned rows and their dependents stay hidden once deletion is
enabled. These check direct parent liveness, not derived-artifact validity,
authorization, or generation barriers. They must not be used as an unlocked
preflight check before a protected write.
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import ColumnElement, and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, aliased
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.selectable import Exists

from src import models

PrimitiveKind = Literal["workspace", "peer", "session", "message"]


@dataclass(frozen=True, slots=True)
class DedicatedTenantContext:
    """An already routed, caller-owned session in one instance's database.

    There is no tenant column in today's schema. This context does not route a
    connection, authorize a caller, or support the shared-pool schema. Shared
    storage must add explicit tenant predicates, including queue/claim scoping,
    before this boundary can be used there. Do not serialize the session into
    durable outbox/purge records.
    """

    db: AsyncSession


@dataclass(frozen=True, slots=True)
class PrimitiveIdentity:
    """Stable identity within a tenant context, never a resource name.

    workspace_uid is Workspace.id. uid is Workspace/Peer/Session.id, or
    Message.public_id (not the integer Message.id used by queue rows).
    Resolve names under the caller's authorization before creating this value.
    """

    kind: PrimitiveKind
    workspace_uid: str
    uid: str

    def __post_init__(self) -> None:
        if self.kind not in ("workspace", "peer", "session", "message"):
            raise ValueError(f"Unsupported primitive kind: {self.kind}")
        if self.kind == "workspace" and self.uid != self.workspace_uid:
            raise ValueError("Workspace identity must match workspace_uid")


def primitive_liveness_clause(identity: PrimitiveIdentity) -> Exists:
    """Build a self-contained EXISTS for one stable identity and its parents.

    All joins use today's dedicated-schema keys. Correlation is disabled: this
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
            workspace.id == identity.workspace_uid,
            workspace.deleted_at.is_(None),
        )
    )

    match identity.kind:
        case "workspace":
            pass  # PrimitiveIdentity requires uid == workspace_uid.
        case "peer":
            stmt = stmt.join(
                models.Peer, models.Peer.workspace_name == workspace.name
            ).where(models.Peer.id == identity.uid, models.Peer.deleted_at.is_(None))
        case "session":
            stmt = stmt.join(
                models.Session, models.Session.workspace_name == workspace.name
            ).where(
                models.Session.id == identity.uid,
                models.Session.deleted_at.is_(None),
            )
        case "message":
            stmt = (
                stmt.join(
                    models.Message, models.Message.workspace_name == workspace.name
                )
                .join(
                    models.Session,
                    and_(
                        models.Session.workspace_name == models.Message.workspace_name,
                        models.Session.name == models.Message.session_name,
                    ),
                )
                .join(
                    models.Peer,
                    and_(
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
    context: DedicatedTenantContext, identity: PrimitiveIdentity
) -> bool:
    """Read direct liveness from the database, bypassing caches/ORM identity maps.

    One SELECT, no flush (including autoflush), commit, rollback, or locks. The
    caller owns the transaction and must explicitly flush any pending changes
    it wants this query to see. A normal session can autobegin a transaction;
    the answer reflects that transaction's database snapshot, not future writes.
    This is NOT the serialized final-write barrier required by producers.
    """
    with context.db.no_autoflush:
        return bool(
            await context.db.scalar(select(primitive_liveness_clause(identity)))
        )


def _workspace_live(workspace_name: QueryableAttribute[str]) -> ColumnElement[bool]:
    workspace = aliased(models.Workspace)
    return exists().where(
        workspace.name == workspace_name, workspace.deleted_at.is_(None)
    )


def _peer_live(
    workspace_name: QueryableAttribute[str], peer_name: QueryableAttribute[str]
) -> ColumnElement[bool]:
    peer = aliased(models.Peer)
    return exists().where(
        peer.workspace_name == workspace_name,
        peer.name == peer_name,
        peer.deleted_at.is_(None),
    )


def _session_live(
    workspace_name: QueryableAttribute[str], session_name: QueryableAttribute[str]
) -> ColumnElement[bool]:
    session = aliased(models.Session)
    return exists().where(
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
    return and_(peer.deleted_at.is_(None), _workspace_live(peer.workspace_name))


def live_session_row(
    session: type[models.Session] | AliasedClass[models.Session] = models.Session,
) -> ColumnElement[bool]:
    """Row-wise filter: neither the session nor its workspace is tombstoned.

    ``is_active`` is a separate archive state and is not checked here.
    """
    return and_(
        session.deleted_at.is_(None),
        _workspace_live(session.workspace_name),
    )


def live_message_row(
    message: type[models.Message] | AliasedClass[models.Message] = models.Message,
) -> ColumnElement[bool]:
    """Row-wise filter: the message, its session, author peer and workspace are live."""
    return and_(
        message.deleted_at.is_(None),
        _workspace_live(message.workspace_name),
        _session_live(message.workspace_name, message.session_name),
        _peer_live(message.workspace_name, message.peer_name),
    )


def live_session_peer_row(
    session_peer: type[models.SessionPeer]
    | AliasedClass[models.SessionPeer] = models.SessionPeer,
) -> ColumnElement[bool]:
    """Row-wise filter: the membership's session, peer and workspace are live.

    ``left_at`` is a separate membership state and is not checked here.
    """
    return and_(
        _workspace_live(session_peer.workspace_name),
        _session_live(session_peer.workspace_name, session_peer.session_name),
        _peer_live(session_peer.workspace_name, session_peer.peer_name),
    )
