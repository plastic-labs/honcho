"""Direct liveness only: these tests do not assert deletion is activated."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import aliased

from src import models, schemas
from src.config import settings
from src.crud.lifecycle import (
    PrimitiveIdentity,
    PrimitiveKind,
    TenantContext,
    is_primitive_live,
    primitive_liveness_clause,
)
from src.db import tenant_context

KINDS: tuple[PrimitiveKind, ...] = ("workspace", "peer", "session", "message")


@dataclass
class Graph:
    workspace: models.Workspace
    peer: models.Peer
    session: models.Session
    message: models.Message

    def identity(self, kind: PrimitiveKind) -> PrimitiveIdentity:
        """Use tenant-qualified stable IDs, including public IDs for messages."""
        row = getattr(self, kind)
        uid = row.public_id if kind == "message" else row.id
        return PrimitiveIdentity(kind, self.workspace.id, uid, self.workspace.tenant_id)


async def _seed_graph(
    db: AsyncSession, workspace_name: str, workspace: models.Workspace | None = None
) -> Graph:
    workspace = workspace or models.Workspace(name=workspace_name)
    db.add(workspace)
    await db.flush()
    peer = models.Peer(name="same-peer", workspace_name=workspace_name)
    session = models.Session(name="same-session", workspace_name=workspace_name)
    db.add_all([peer, session])
    await db.flush()
    message = models.Message(
        workspace_name=workspace_name,
        session_name=session.name,
        peer_name=peer.name,
        content="unchanged live content",
        seq_in_session=1,
    )
    db.add(message)
    await db.flush()
    return Graph(workspace, peer, session, message)


@pytest.mark.parametrize("kind", KINDS)
async def test_direct_tombstone_and_live_schema_contract(
    db_session: AsyncSession, kind: PrimitiveKind
) -> None:
    """Hide direct tombstones without exposing lifecycle state in public schemas."""
    graph = await _seed_graph(db_session, "direct")
    context = TenantContext(db_session)
    identity = graph.identity(kind)
    row = getattr(graph, kind)
    assert row.deleted_at is None
    assert await is_primitive_live(context, identity)

    # No public schema change or lifecycle activation accompanies the ORM column.
    api_schema = {
        "workspace": schemas.Workspace,
        "peer": schemas.Peer,
        "session": schemas.Session,
        "message": schemas.Message,
    }[kind]
    assert "deleted_at" not in api_schema.model_fields

    row.deleted_at = datetime.now(UTC)
    await db_session.flush()
    assert not await is_primitive_live(context, identity)


@pytest.mark.parametrize(
    ("parent", "hidden"),
    [
        ("workspace", {"workspace", "peer", "session", "message"}),
        ("session", {"session", "message"}),
        ("peer", {"peer", "message"}),
    ],
)
async def test_parent_tombstones_and_workspace_isolation(
    db_session: AsyncSession, parent: PrimitiveKind, hidden: set[str]
) -> None:
    """Hide only direct dependents of a tombstoned parent within its workspace."""
    target = await _seed_graph(db_session, "target")
    other = await _seed_graph(db_session, "other")
    context = TenantContext(db_session)
    getattr(target, parent).deleted_at = datetime.now(UTC)
    await db_session.flush()
    for kind in KINDS:
        assert await is_primitive_live(context, target.identity(kind)) == (
            kind not in hidden
        )
        assert await is_primitive_live(context, other.identity(kind))
    if parent == "peer":
        # A deleted author does not hide another author's message in the same
        # session, or turn the session into a tombstone.
        bystander = models.Peer(name="bystander", workspace_name=target.workspace.name)
        db_session.add(bystander)
        await db_session.flush()
        message = models.Message(
            workspace_name=target.workspace.name,
            session_name=target.session.name,
            peer_name=bystander.name,
            content="other author",
            seq_in_session=2,
        )
        db_session.add(message)
        await db_session.flush()
        assert await is_primitive_live(
            context,
            PrimitiveIdentity(
                "message",
                target.workspace.id,
                message.public_id,
                target.workspace.tenant_id,
            ),
        )
    assert context.db is db_session


async def test_archive_and_membership_state_are_not_tombstones(
    db_session: AsyncSession,
) -> None:
    """Keep archival and membership departure independent of primitive liveness."""
    graph = await _seed_graph(db_session, "archived")
    graph.session.is_active = False
    db_session.add(
        models.SessionPeer(
            workspace_name=graph.workspace.name,
            session_name=graph.session.name,
            peer_name=graph.peer.name,
            left_at=datetime.now(UTC),
        )
    )
    await db_session.flush()
    for kind in KINDS:
        assert await is_primitive_live(TenantContext(db_session), graph.identity(kind))


async def test_stable_identity_not_names_or_numeric_message_ids(
    db_session: AsyncSession,
) -> None:
    """Keep stable identities distinct from reused names and numeric message IDs."""
    graph = await _seed_graph(db_session, "identities")
    other = await _seed_graph(db_session, "other-identities")
    context = TenantContext(db_session)
    for kind in KINDS:
        row = getattr(graph, kind)
        unsafe_uid = str(row.id) if kind == "message" else row.name
        workspace_uid = unsafe_uid if kind == "workspace" else graph.workspace.id
        assert not await is_primitive_live(
            context,
            PrimitiveIdentity(
                kind, workspace_uid, unsafe_uid, graph.workspace.tenant_id
            ),
        )
        assert not await is_primitive_live(
            context,
            PrimitiveIdentity(kind, "missing", "missing", graph.workspace.tenant_id),
        )
        if kind != "workspace":
            assert not await is_primitive_live(
                context,
                PrimitiveIdentity(
                    kind,
                    other.workspace.id,
                    graph.identity(kind).uid,
                    other.workspace.tenant_id,
                ),
            )

    # Peer/session name reuse within the SAME live workspace must not retarget
    # their old identities. Replacing the workspace alone cannot prove this.
    child_identities = [graph.identity(kind) for kind in KINDS if kind != "workspace"]
    for row in (graph.message, graph.session, graph.peer):
        await db_session.delete(row)
        await db_session.flush()
    graph = await _seed_graph(db_session, "identities", graph.workspace)
    for identity in child_identities:
        assert not await is_primitive_live(context, identity)
        assert await is_primitive_live(context, graph.identity(identity.kind))

    old_identities = [graph.identity(kind) for kind in KINDS]
    for row in (graph.message, graph.session, graph.peer, graph.workspace):
        await db_session.delete(row)
        await db_session.flush()
    replacement = await _seed_graph(db_session, "identities")
    for identity in old_identities:
        assert not await is_primitive_live(context, identity)
        # Same workspace NAME cannot retarget an old stable workspace identity.
        if identity.kind != "workspace":
            assert not await is_primitive_live(
                context,
                PrimitiveIdentity(
                    identity.kind,
                    identity.workspace_uid,
                    replacement.identity(identity.kind).uid,
                    identity.tenant_id,
                ),
            )
        assert await is_primitive_live(context, replacement.identity(identity.kind))


async def test_identity_clause_does_not_correlate_with_outer_tables(
    db_session: AsyncSession,
) -> None:
    live = await _seed_graph(db_session, "live")
    dead = await _seed_graph(db_session, "dead")
    dead.workspace.deleted_at = datetime.now(UTC)
    await db_session.flush()
    outer = aliased(models.Workspace)
    # The selected identity is independent of both the alias and unaliased
    # Workspace in the enclosing query, even when that outer row is tombstoned.
    stmt = (
        select(primitive_liveness_clause(live.identity("message")))
        .select_from(outer)
        .join(models.Workspace, models.Workspace.id == outer.id)
        .where(outer.id == dead.workspace.id)
    )
    assert await db_session.scalar(stmt) is True


async def test_caller_owns_flush_commit_rollback_and_snapshot(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """Keep authoritative liveness reads inside the caller's transaction contract."""
    graph = await _seed_graph(db_session, "transactions")
    await db_session.commit()
    identity = graph.identity("message")
    message_key = (graph.message.tenant_id, graph.message.id)
    context = TenantContext(db_session)

    # A normal autoflush-enabled session must not flush dirty or pending rows.
    graph.message.deleted_at = datetime.now(UTC)
    pending = models.Workspace(name="pending")
    db_session.add(pending)
    transaction = db_session.get_transaction()
    assert db_session.autoflush
    assert await is_primitive_live(context, identity)
    assert graph.message in db_session.dirty
    assert pending in db_session.new
    assert db_session.get_transaction() is transaction

    await db_session.flush()
    assert not await is_primitive_live(context, identity)
    # The helper did not commit or roll back the caller's flushed tombstone.
    async with AsyncSession(db_engine) as reader:
        assert await is_primitive_live(TenantContext(reader), identity)
    await db_session.rollback()
    assert await is_primitive_live(context, identity)

    # Bypass a stale ORM instance: liveness reads the authoritative database.
    cached_message = await db_session.get(models.Message, message_key)
    assert cached_message is not None and cached_message.deleted_at is None
    async with AsyncSession(db_engine) as writer:
        await writer.execute(
            update(models.Message)
            .where(models.Message.public_id == identity.uid)
            .values(deleted_at=datetime.now(UTC))
        )
        await writer.commit()
    assert cached_message.deleted_at is None
    assert not await is_primitive_live(context, identity)


def test_workspace_identity_cannot_contradict_its_scope() -> None:
    """Reject a workspace identity that contradicts its stable workspace scope."""
    with pytest.raises(ValueError, match="must match workspace_uid"):
        PrimitiveIdentity("workspace", "workspace-a", "workspace-b", "default")


@pytest.mark.parametrize("tenant_id", [None, "dedicated-instance"])
async def test_dedicated_context_defaults_to_instance_tenant(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tenant_id: str | None
) -> None:
    """Use the instance tenant rather than an unrelated ambient tenant."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings, "TENANT_ID", tenant_id)
    token = tenant_context.set("unrelated-ambient-tenant")
    try:
        context = TenantContext(db_session)
        assert context.tenant_id == (tenant_id or "default")
        assert context.db is db_session
    finally:
        tenant_context.reset(token)


async def test_shared_context_captures_bound_tenant_and_fails_closed(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture the bound tenant and refuse tenant-less shared-storage contexts."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="no tenant bound"):
            TenantContext(db_session)
        assert (
            TenantContext(db_session, "explicit-service-target").tenant_id
            == "explicit-service-target"
        )
        tenant_context.set("tenant-a")
        context = TenantContext(db_session)
        tenant_context.set("tenant-b")
        assert context.tenant_id == "tenant-a"
    finally:
        tenant_context.reset(token)


async def test_context_identity_mismatch_does_not_touch_transaction(
    db_session: AsyncSession,
) -> None:
    """Reject tenant mismatches without touching the caller's transaction."""
    context = TenantContext(db_session, "tenant-a")
    identity = PrimitiveIdentity("message", "workspace", "message", "tenant-b")
    transaction = db_session.get_transaction()
    with pytest.raises(ValueError, match="context tenant"):
        await is_primitive_live(context, identity)
    assert db_session.get_transaction() is transaction


async def test_empty_tenant_identity_and_context_are_rejected(
    db_session: AsyncSession,
) -> None:
    """Reject empty tenant IDs before any scoped liveness check can run."""
    with pytest.raises(ValueError, match="requires a tenant_id"):
        TenantContext(db_session, "")
    with pytest.raises(ValueError, match="requires a tenant_id"):
        PrimitiveIdentity("workspace", "workspace", "workspace", "")
