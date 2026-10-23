"""Inactive deletion foundation for today's dedicated schema.

No production read/delete path uses these helpers yet. They check direct parent
liveness, not derived-artifact validity, authorization, or generation barriers.
They must not be used as an unlocked preflight check before a protected write.
"""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
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
