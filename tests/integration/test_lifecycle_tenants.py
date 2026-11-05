"""Deletion liveness on the non-owner service role that bypasses real RLS."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import aliased

import src.utils.search as search_utils
from src import crud, models, schemas
from src.config import settings
from src.crud.lifecycle import (
    PrimitiveIdentity,
    PrimitiveKind,
    TenantContext,
    is_primitive_live,
    live_document_row,
    live_message_row,
    live_peer_row,
    live_session_peer_row,
    live_session_row,
    live_workspace_row,
    primitive_liveness_clause,
)
from src.crud.message import get_peer_session_names, observer_scope_clause
from src.db import CROSS_TENANT, tenant_context
from src.deriver.enqueue import get_peers_with_configuration
from src.utils import agent_tools
from src.utils.agent_tools import ToolContext, get_observation_context, get_tool_handler
from src.utils.search import search
from tests.integration.lifecycle_fixtures import migrated_lifecycle_engine


class CollectionPeers(TypedDict):
    observer: str
    observed: str


@pytest.fixture(scope="session")
def lifecycle_engine(worker_id: str) -> Iterator[Engine]:
    """A separately migrated RLS database with the shared-pool natural keys.

    The migrated dedicated schema keeps a tenant-less compatibility index for
    the previous release's upsert. Shared storage uses the tenant-qualified key.
    Remove that index only in this throwaway fixture, never in the application.
    """
    with migrated_lifecycle_engine(worker_id) as engine:
        yield engine


@pytest.fixture(scope="session")
def lifecycle_service_role(lifecycle_engine: Engine) -> Iterator[str]:
    """Provide a non-superuser BYPASSRLS role for tenant-collision probes."""
    role = "lifecycle_service_" + uuid4().hex
    with lifecycle_engine.begin() as conn:
        conn.execute(text(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER BYPASSRLS"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))
        conn.execute(text(f"GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO {role}"))
    try:
        yield role
    finally:
        with lifecycle_engine.begin() as conn:
            conn.execute(text(f"DROP OWNED BY {role}"))
            conn.execute(text(f"DROP ROLE {role}"))


@pytest.fixture
async def lifecycle_service_db(
    lifecycle_engine: Engine,
) -> AsyncIterator[AsyncSession]:
    """Yield an isolated session for explicit cross-tenant service probes."""
    engine = create_async_engine(lifecycle_engine.url)
    try:
        async with AsyncSession(engine, info={CROSS_TENANT: True}) as db:
            yield db
            await db.rollback()
    finally:
        await engine.dispose()


async def _seed_graphs(
    db: AsyncSession,
) -> tuple[
    dict[PrimitiveKind, PrimitiveIdentity], dict[PrimitiveKind, PrimitiveIdentity]
]:
    """Seed two tenants with colliding names and stable primitive identifiers."""
    ids = {
        kind: generate_nanoid() for kind in ("workspace", "peer", "session", "message")
    }
    identities: list[dict[PrimitiveKind, PrimitiveIdentity]] = []
    for tenant_id in ("lifecycle-a", "lifecycle-b"):
        db.add(models.Tenant(tenant_id=tenant_id))
        await db.flush()
        db.add(
            models.Workspace(tenant_id=tenant_id, id=ids["workspace"], name="same-ws")
        )
        await db.flush()
        db.add_all(
            [
                models.Peer(
                    tenant_id=tenant_id,
                    id=ids["peer"],
                    workspace_name="same-ws",
                    name="same-peer",
                ),
                models.Session(
                    tenant_id=tenant_id,
                    id=ids["session"],
                    workspace_name="same-ws",
                    name="same-session",
                    is_active=False,
                ),
            ]
        )
        await db.flush()
        db.add_all(
            [
                models.Message(
                    tenant_id=tenant_id,
                    public_id=ids["message"],
                    workspace_name="same-ws",
                    session_name="same-session",
                    peer_name="same-peer",
                    seq_in_session=1,
                    content="same content",
                ),
                models.SessionPeer(
                    tenant_id=tenant_id,
                    workspace_name="same-ws",
                    session_name="same-session",
                    peer_name="same-peer",
                    left_at=datetime.now(UTC),
                ),
            ]
        )
        await db.flush()
        identities.append(
            {
                kind: PrimitiveIdentity(kind, ids["workspace"], ids[kind], tenant_id)
                for kind in ("workspace", "peer", "session", "message")
            }
        )
    return identities[0], identities[1]


@pytest.mark.parametrize(
    ("parent", "hidden"),
    [
        ("workspace", {"workspace", "peer", "session", "message", "membership"}),
        ("peer", {"peer", "message", "membership"}),
        ("session", {"session", "message", "membership"}),
        ("message", {"message"}),
    ],
)
@pytest.mark.parametrize("use_alias", [False, True])
async def test_same_names_and_ids_stay_in_their_tenant_on_service_role(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
    parent: PrimitiveKind,
    hidden: set[str],
    use_alias: bool,
) -> None:
    """Keep colliding identities isolated when real service reads bypass RLS."""
    db = lifecycle_service_db
    target, other = await _seed_graphs(db)
    primitive_models = {
        "workspace": models.Workspace,
        "peer": models.Peer,
        "session": models.Session,
        "message": models.Message,
    }
    model = primitive_models[parent]
    row = await db.scalar(select(model).where(model.tenant_id == "lifecycle-a"))
    assert row is not None
    row.deleted_at = datetime.now(UTC)
    await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    role = (
        await db.execute(
            text(
                "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
            )
        )
    ).one()
    assert role == (True, False)
    assert await db.scalar(
        text(
            "SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE relname = 'workspaces'"
        )
    )
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    for tenant, identities in (("lifecycle-a", target), ("lifecycle-b", other)):
        context = TenantContext(db, tenant)
        for kind, identity in identities.items():
            expected = tenant == "lifecycle-b" or kind not in hidden
            assert await is_primitive_live(context, identity) is expected
            assert (
                await db.scalar(select(primitive_liveness_clause(identity))) is expected
            )

    for model, clause, kind in (
        (models.Workspace, live_workspace_row, "workspace"),
        (models.Peer, live_peer_row, "peer"),
        (models.Session, live_session_row, "session"),
        (models.Message, live_message_row, "message"),
        (models.SessionPeer, live_session_peer_row, "membership"),
    ):
        entity = aliased(model) if use_alias else model
        tenants = set(
            await db.scalars(
                select(entity.tenant_id)
                .where(
                    entity.workspace_name == "same-ws"
                    if kind != "workspace"
                    else entity.name == "same-ws"
                )
                .where(clause(entity))
            )
        )
        assert tenants == (
            {"lifecycle-b"} if kind in hidden else {"lifecycle-a", "lifecycle-b"}
        )

    with pytest.raises(ValueError, match="context tenant"):
        await is_primitive_live(TenantContext(db, "lifecycle-b"), target["message"])


async def test_liveness_exists_uses_tenant_led_parent_indexes(
    lifecycle_service_db: AsyncSession, lifecycle_service_role: str
) -> None:
    """Retain positive tenant-led parent lookups in liveness execution plans."""
    db = lifecycle_service_db
    await _seed_graphs(db)
    db.add_all(
        model(tenant_id="lifecycle-a", workspace_name="same-ws", name=f"noise-{i}")
        for model in (models.Peer, models.Session)
        for i in range(200)
    )
    await db.flush()
    for table in ("workspaces", "peers", "sessions", "messages"):
        await db.execute(text(f"ANALYZE public.{table}"))
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    await db.execute(text("SET LOCAL enable_seqscan = off"))
    stmt = select(models.Message.public_id).where(
        models.Message.tenant_id == "lifecycle-a", live_message_row()
    )
    sql = str(
        stmt.compile(dialect=db.bind.dialect, compile_kwargs={"literal_binds": True})
    )
    assert "NOT (EXISTS" not in sql
    assert sql.count("EXISTS") == 3
    plan = "\n".join(await db.scalars(text("EXPLAIN " + sql)))
    assert "uq_workspaces_tenant_id_name" in plan, plan
    assert "tenant_id" in plan
    assert "Seq Scan" not in plan, plan
    assert "Index Scan" in plan or "Index Only Scan" in plan, plan

    projected = (
        select(live_message_row())
        .select_from(models.Message)
        .where(models.Message.tenant_id == "lifecycle-a")
    )
    projected_sql = str(
        projected.compile(
            dialect=db.bind.dialect, compile_kwargs={"literal_binds": True}
        )
    )
    projected_plan = "\n".join(await db.scalars(text("EXPLAIN " + projected_sql)))
    for index in (
        "uq_workspaces_tenant_id_name",
        "uq_peers_tenant_id_name_workspace_name",
        "uq_sessions_tenant_id_name_workspace_name",
    ):
        assert index in projected_plan, projected_plan


async def _seed_conclusions(db: AsyncSession) -> dict[str, str]:
    """Create colliding conclusion IDs in both directions for two tenants."""
    await _seed_graphs(db)
    ids = {kind: generate_nanoid() for kind in ("session", "sessionless")}
    for tenant in ("lifecycle-a", "lifecycle-b"):
        db.add(
            models.Peer(tenant_id=tenant, workspace_name="same-ws", name="other-peer")
        )
        await db.flush()
        for observer, observed in (
            ("same-peer", "other-peer"),
            ("other-peer", "same-peer"),
        ):
            db.add(
                models.Collection(
                    tenant_id=tenant,
                    workspace_name="same-ws",
                    observer=observer,
                    observed=observed,
                )
            )
            await db.flush()
            db.add_all(
                models.Document(
                    tenant_id=tenant,
                    id=ids[kind] if observer == "same-peer" else generate_nanoid(),
                    workspace_name="same-ws",
                    observer=observer,
                    observed=observed,
                    session_name="same-session" if kind == "session" else None,
                    content=kind,
                    level="explicit",
                    embedding=[1.0]
                    + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 1),
                )
                for kind in ids
            )
            await db.flush()
    return ids


@pytest.mark.parametrize("parent", ["none", "workspace", "peer", "session", "document"])
@pytest.mark.parametrize("use_alias", [False, True])
async def test_conclusion_liveness_and_reads_pin_tombstone_sets_to_tenant(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
    use_alias: bool,
) -> None:
    """Pin conclusion tombstone sets to the target tenant on BYPASSRLS reads."""
    db = lifecycle_service_db
    ids = await _seed_conclusions(db)
    if parent != "none":
        model = {
            "workspace": models.Workspace,
            "peer": models.Peer,
            "session": models.Session,
            "document": models.Document,
        }[parent]
        stmt = select(model).where(model.tenant_id == "lifecycle-a")
        if parent == "peer":
            stmt = stmt.where(model.name == "same-peer")
        if parent == "document":
            stmt = stmt.where(model.id == ids["session"])
        row = await db.scalar(stmt)
        assert row is not None
        row.deleted_at = datetime.now(UTC)
        await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    assert (
        await db.execute(
            text(
                "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
            )
        )
    ).one() == (True, False)
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    entity = aliased(models.Document) if use_alias else models.Document
    for tenant in ("lifecycle-a", "lifecycle-b"):
        expected: set[str] = set(ids.values())
        if tenant == "lifecycle-a":
            if parent in ("workspace", "peer"):
                expected = set()
            elif parent in ("session", "document"):
                expected.remove(ids["session"])
        rows = list(
            await db.scalars(
                select(entity)
                .join(
                    models.Workspace,
                    (models.Workspace.tenant_id == entity.tenant_id)
                    & (models.Workspace.name == entity.workspace_name),
                )
                .where(live_document_row("same-ws", entity, tenant_id=tenant))
            )
        )
        assert {doc.tenant_id for doc in rows} <= {tenant}
        forward = [doc for doc in rows if doc.observer == "same-peer"]
        assert {doc.id for doc in forward} == expected
        reverse = [doc for doc in rows if doc.observer == "other-peer"]
        assert len(reverse) == (
            0
            if tenant == "lifecycle-a" and parent in ("workspace", "peer")
            else 1
            if tenant == "lifecycle-a" and parent == "session"
            else 2
        )
        token = tenant_context.set(tenant)
        try:
            kwargs: CollectionPeers = {
                "observer": "same-peer",
                "observed": "other-peer",
            }
            reads = [
                list(await db.scalars(crud.get_all_documents("same-ws", **kwargs))),
                list(
                    await db.scalars(
                        crud.get_documents_with_filters(
                            "same-ws", filters={"observer": "same-peer"}
                        )
                    )
                ),
                await crud.get_documents_by_ids(db, "same-ws", list(ids.values())),
                await crud.fetch_documents_by_ids(
                    db, "same-ws", document_ids=list(ids.values()), **kwargs
                ),
                await crud.query_documents_recent(db, "same-ws", **kwargs),
                await crud.query_documents_most_derived(db, "same-ws", **kwargs),
                await crud.query_documents(
                    db,
                    "same-ws",
                    "anything",
                    top_k=10,
                    embedding=[1.0]
                    + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 1),
                    **kwargs,
                ),
            ]
            explicit, total = await crud.query_explicit_documents_after(
                db, "same-ws", after=None, since=None, limit=10, **kwargs
            )
            reads.append(explicit)
            assert total == len(expected)
            summary = await crud.summarize_documents_by_level(db, "same-ws", **kwargs)
            assert sum(summary.counts_by_level.values()) == len(expected)
            for documents in reads:
                assert {doc.id for doc in documents} == expected
                assert {doc.tenant_id for doc in documents} <= {tenant}
        finally:
            tenant_context.reset(token)


async def test_conclusion_tombstone_subplans_are_uncorrelated_and_tenant_pinned(
    lifecycle_service_db: AsyncSession, lifecycle_service_role: str
) -> None:
    """Keep conclusion tombstone subplans uncorrelated and tenant-qualified."""
    db = lifecycle_service_db
    await _seed_conclusions(db)
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    await db.execute(text("SET LOCAL enable_seqscan = off"))
    stmt = select(models.Document.id).where(
        live_document_row("same-ws", tenant_id="lifecycle-a")
    )
    sql = str(
        stmt.compile(dialect=db.bind.dialect, compile_kwargs={"literal_binds": True})
    )
    assert "NOT IN (SELECT" in sql
    assert "NOT (EXISTS" not in sql
    assert sql.count("tenant_id = 'lifecycle-a'") == 5
    plan = "\n".join(await db.scalars(text("EXPLAIN (ANALYZE, BUFFERS) " + sql)))
    assert "hashed SubPlan" in plan, plan
    assert "InitPlan" in plan, plan
    assert "uq_workspaces_tenant_id_name" in plan, plan
    assert "tenant_id" in plan, plan


async def test_observation_context_neighbours_cannot_cross_service_tenants(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent same-name co-tenant messages from becoming context neighbours."""
    db = lifecycle_service_db
    await _seed_graphs(db)
    target_id = generate_nanoid()
    db.add(
        models.Message(
            tenant_id="lifecycle-a",
            public_id=target_id,
            workspace_name="same-ws",
            session_name="same-session",
            peer_name="same-peer",
            seq_in_session=2,
            content="only A target",
        )
    )
    await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    rows = await get_observation_context(db, "same-ws", None, [target_id])
    assert len(rows) == 2
    assert {row.tenant_id for row in rows} == {"lifecycle-a"}

    @asynccontextmanager
    async def service_reader(
        *_args: object, **_kwargs: object
    ) -> AsyncGenerator[AsyncSession]:
        yield db

    monkeypatch.setattr("src.crud.message.tracked_db", service_reader)
    snippets = await crud.grep_messages("same-ws", None, "only A target")
    assert len(snippets) == 1
    assert {row.tenant_id for _, context in snippets for row in context} == {
        "lifecycle-a"
    }


async def test_semantic_message_join_cannot_use_another_tenants_embedding(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep message-embedding joins tenant-qualified on real service reads."""
    db = lifecycle_service_db
    target, _ = await _seed_graphs(db)
    embedding = [1.0] + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 1)
    db.add(
        models.MessageEmbedding(
            tenant_id="lifecycle-a",
            workspace_name="same-ws",
            session_name="same-session",
            peer_name="same-peer",
            message_id=target["message"].uid,
            content="A embedding",
            embedding=embedding,
            sync_state="synced",
        )
    )
    await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))

    @asynccontextmanager
    async def service_reader(
        *_args: object, **_kwargs: object
    ) -> AsyncGenerator[AsyncSession]:
        yield db

    monkeypatch.setattr("src.utils.search.tracked_db", service_reader)
    monkeypatch.setattr("src.crud.message.tracked_db", service_reader)
    monkeypatch.setattr(
        "src.utils.search.embedding_client.embed", AsyncMock(return_value=embedding)
    )
    monkeypatch.setattr(settings, "EMBED_MESSAGES", True)
    rows = await search("no lexical match", filters={"workspace_id": "same-ws"})
    assert len(rows) == 1
    assert {row.tenant_id for row in rows} == {"lifecycle-a"}
    snippets = await crud.search_messages(
        "same-ws", None, "anything", embedding=embedding
    )
    assert len(snippets) == 1
    assert {row.tenant_id for matches, _ in snippets for row in matches} == {
        "lifecycle-a"
    }
    assert {row.tenant_id for _, context in snippets for row in context} == {
        "lifecycle-a"
    }


@pytest.mark.parametrize("parent", ["none", "workspace", "observer", "session"])
@pytest.mark.parametrize("left", [False, True])
async def test_internal_membership_reads_require_live_parents_on_service_role(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
    left: bool,
) -> None:
    """Preserve tenant-scoped history without trusting tombstoned membership parents."""
    db = lifecycle_service_db
    await _seed_graphs(db)
    instant = datetime(2026, 1, 2, tzinfo=UTC)
    for tenant in ("lifecycle-a", "lifecycle-b"):
        peer = await db.scalar(
            select(models.Peer).where(models.Peer.tenant_id == tenant)
        )
        assert peer is not None
        peer.configuration = {"observe_me": tenant == "lifecycle-a"}
        membership = await db.scalar(
            select(models.SessionPeer).where(models.SessionPeer.tenant_id == tenant)
        )
        assert membership is not None
        membership.joined_at = instant
        membership.left_at = instant + timedelta(days=2) if left else None
        db.add(models.Peer(tenant_id=tenant, workspace_name="same-ws", name="author"))
        await db.flush()
        db.add_all(
            models.Message(
                tenant_id=tenant,
                workspace_name="same-ws",
                session_name="same-session",
                peer_name="author",
                seq_in_session=seq,
                created_at=instant + timedelta(days=offset),
                content="perspective sentinel",
            )
            for seq, offset in ((2, -1), (3, 0), (4, 1), (5, 2), (6, 3))
        )
        db.add(
            models.Session(tenant_id=tenant, workspace_name="same-ws", name="unrelated")
        )
        db.add(models.Workspace(tenant_id=tenant, name="other-ws"))
        await db.flush()
        db.add_all(
            [
                models.Peer(tenant_id=tenant, workspace_name="other-ws", name="author"),
                models.Session(
                    tenant_id=tenant, workspace_name="other-ws", name="same-session"
                ),
            ]
        )
        await db.flush()
        db.add(
            models.Message(
                tenant_id=tenant,
                workspace_name="other-ws",
                session_name="same-session",
                peer_name="author",
                seq_in_session=42,
                created_at=instant + timedelta(days=1),
                content="perspective sentinel",
            )
        )
    await db.flush()
    if parent != "none":
        model = {
            "workspace": models.Workspace,
            "observer": models.Peer,
            "session": models.Session,
        }[parent]
        stmt = select(model).where(model.tenant_id == "lifecycle-a")
        if parent == "workspace":
            stmt = stmt.where(models.Workspace.name == "same-ws")
        else:
            stmt = stmt.where(model.workspace_name == "same-ws")
        if parent == "observer":
            stmt = stmt.where(model.name == "same-peer")
        if parent == "session":
            stmt = stmt.where(model.name == "same-session")
        row = await db.scalar(stmt)
        assert row is not None
        row.deleted_at = datetime.now(UTC)
        await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    assert (
        await db.execute(
            text(
                "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
            )
        )
    ).one() == (True, False)
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings, "EMBED_MESSAGES", False)

    @asynccontextmanager
    async def service_reader(
        *_args: object, **_kwargs: object
    ) -> AsyncGenerator[AsyncSession]:
        yield db

    monkeypatch.setattr("src.utils.search.tracked_db", service_reader)
    for tenant in ("lifecycle-a", "lifecycle-b"):
        token = tenant_context.set(tenant)
        try:
            visible = tenant == "lifecycle-b" or parent == "none"
            configuration = await get_peers_with_configuration(
                db, "same-ws", "same-session"
            )
            assert configuration == (
                {"same-peer": [{"observe_me": tenant == "lifecycle-a"}, {}, not left]}
                if visible
                else {}
            )
            names = await get_peer_session_names(db, "same-ws", "same-peer")
            core_names = list(
                await db.scalars(
                    select(models.Session.name).where(
                        models.Session.tenant_id == tenant,
                        models.Session.workspace_name == "same-ws",
                        observer_scope_clause(
                            "same-ws", "same-peer", models.Session.name
                        ),
                    )
                )
            )
            assert core_names == names == (["same-session"] if visible else [])
            assert await get_peer_session_names(
                db, "same-ws", "same-peer", active_only=True
            ) == (["same-session"] if visible and not left else [])
            messages = list(
                await db.scalars(
                    select(models.Message).where(models.Message.peer_name == "author")
                )
            )
            filtered = await vars(search_utils)["_filter_by_peer_perspective"](
                db, messages, "same-ws", "same-peer"
            )
            expected = {3, 4, 5} if left else {3, 4, 5, 6}
            assert {msg.seq_in_session for msg in filtered} == (
                expected if visible else set()
            )
            assert {msg.tenant_id for msg in filtered} <= {tenant}
            await db.execute(
                text("SELECT set_config('app.tenant', :tenant, true)"),
                {"tenant": tenant},
            )
            lexical = await search(
                "perspective sentinel",
                filters={"workspace_id": "same-ws", "peer_perspective": "same-peer"},
            )
            assert {msg.seq_in_session for msg in lexical} == (
                expected if visible else set()
            )
            assert {msg.tenant_id for msg in lexical} <= {tenant}
            if parent == "observer":
                surviving = list(
                    await db.scalars(
                        select(models.Message).where(
                            models.Message.tenant_id == tenant,
                            models.Message.workspace_name == "same-ws",
                            models.Message.peer_name == "author",
                            live_message_row(),
                        )
                    )
                )
                assert len(surviving) == 5
        finally:
            tenant_context.reset(token)


@pytest.mark.parametrize(
    "parent", ["none", "workspace", "observer", "observed", "session", "document"]
)
async def test_internal_conclusion_grounding_and_chain_reads_on_service_role(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
) -> None:
    """Require live parents for grounded sources and fetched reasoning-chain content."""
    db = lifecycle_service_db
    ids = await _seed_conclusions(db)
    timestamp = "2026-01-02T00:00:00Z"
    for tenant in ("lifecycle-a", "lifecycle-b"):
        source = await db.scalar(
            select(models.Document).where(
                models.Document.tenant_id == tenant,
                models.Document.id == ids["session"],
            )
        )
        assert source is not None
        source.internal_metadata = {"message_created_at": timestamp}
        derived = await db.scalar(
            select(models.Document).where(
                models.Document.tenant_id == tenant,
                models.Document.id == ids["sessionless"],
            )
        )
        assert derived is not None
        derived.level = "deductive"
        derived.source_ids = [ids["session"]]
    await db.flush()
    if parent != "none":
        model = {
            "workspace": models.Workspace,
            "observer": models.Peer,
            "observed": models.Peer,
            "session": models.Session,
            "document": models.Document,
        }[parent]
        stmt = select(model).where(model.tenant_id == "lifecycle-a")
        if parent in ("observer", "observed"):
            stmt = stmt.where(
                model.name == ("same-peer" if parent == "observer" else "other-peer")
            )
        if parent == "document":
            stmt = stmt.where(model.id == ids["session"])
        row = await db.scalar(stmt)
        assert row is not None
        row.deleted_at = datetime.now(UTC)
        await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    assert (
        await db.execute(
            text(
                "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
            )
        )
    ).one() == (True, False)
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    @asynccontextmanager
    async def service_reader(
        *_args: object, **_kwargs: object
    ) -> AsyncGenerator[AsyncSession]:
        yield db

    monkeypatch.setattr(agent_tools, "tracked_db", service_reader)
    observation = schemas.ObservationInput(
        content="Derived sentinel",
        level="deductive",
        source_ids=[ids["session"]],
        premises=["Source sentinel"],
    )
    for tenant in ("lifecycle-a", "lifecycle-b"):
        token = tenant_context.set(tenant)
        try:
            visible = tenant == "lifecycle-b" or parent == "none"
            ctx = ToolContext(
                workspace_name="same-ws",
                observer="same-peer",
                observed="other-peer",
                session_name=None,
                current_messages=None,
                include_observation_ids=False,
                db_lock=asyncio.Lock(),
            )
            grounded, failures = await vars(agent_tools)[
                "_filter_ungrounded_source_ids"
            ](
                [observation],
                workspace_name="same-ws",
                observer="same-peer",
                observed="other-peer",
            )
            assert bool(grounded) is visible
            assert bool(failures) is not visible
            assert await vars(agent_tools)["_latest_source_timestamp"](
                ctx, [observation]
            ) == (timestamp if visible else None)
            handler = get_tool_handler("get_reasoning_chain")
            assert handler is not None
            chain = await handler(ctx, {"observation_id": ids["session"]})
            assert ("not found" not in chain) is visible
            derived_chain = await handler(ctx, {"observation_id": ids["sessionless"]})
            assert (f" - [id:{ids['session']}]" in derived_chain) is visible
            foreign_scope = await handler(
                replace(ctx, observer="scope.private"),
                {"observation_id": ids["sessionless"]},
            )
            assert "not found" in foreign_scope
            restricted = await handler(
                replace(ctx, session_allowlist=["same-session"]),
                {"observation_id": ids["session"]},
            )
            assert "unavailable for session-scoped" in restricted
            (
                wrong_collection,
                failures,
            ) = await vars(agent_tools)["_filter_ungrounded_source_ids"](
                [observation],
                workspace_name="same-ws",
                observer="other-peer",
                observed="same-peer",
            )
            assert wrong_collection == []
            assert len(failures) == 1
        finally:
            tenant_context.reset(token)


@pytest.mark.parametrize("parent", ["none", "workspace", "observer", "session"])
@pytest.mark.parametrize("left", [False, True])
async def test_named_scope_sql_membership_stays_active_and_live(
    lifecycle_service_db: AsyncSession,
    lifecycle_service_role: str,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
    left: bool,
) -> None:
    """Require live parents and active named-scope membership in SQL and ORM reads."""
    db = lifecycle_service_db
    await _seed_graphs(db)
    for tenant in ("lifecycle-a", "lifecycle-b"):
        db.add(
            models.Peer(
                tenant_id=tenant,
                workspace_name="same-ws",
                name="scope.private",
                internal_metadata={"kind": "scope"},
            )
        )
        await db.flush()
        db.add(
            models.SessionPeer(
                tenant_id=tenant,
                workspace_name="same-ws",
                session_name="same-session",
                peer_name="scope.private",
                left_at=datetime.now(UTC) if left else None,
            )
        )
    await db.flush()
    if parent != "none":
        model = {
            "workspace": models.Workspace,
            "observer": models.Peer,
            "session": models.Session,
        }[parent]
        stmt = select(model).where(model.tenant_id == "lifecycle-a")
        if parent == "observer":
            stmt = stmt.where(model.name == "scope.private")
        row = await db.scalar(stmt)
        assert row is not None
        row.deleted_at = datetime.now(UTC)
        await db.flush()
    await db.execute(text(f"SET LOCAL ROLE {lifecycle_service_role}"))
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    for tenant in ("lifecycle-a", "lifecycle-b"):
        token = tenant_context.set(tenant)
        try:
            names = await get_peer_session_names(
                db, "same-ws", "scope.private", active_only=True
            )
            sql_names = list(
                await db.scalars(
                    select(models.Session.name).where(
                        models.Session.tenant_id == tenant,
                        models.Session.workspace_name == "same-ws",
                        observer_scope_clause(
                            "same-ws", "scope.private", models.Session.name
                        ),
                    )
                )
            )
            assert (
                names
                == sql_names
                == (
                    ["same-session"]
                    if not left and (tenant == "lifecycle-b" or parent == "none")
                    else []
                )
            )
        finally:
            tenant_context.reset(token)
