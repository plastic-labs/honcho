"""Deletion liveness on the non-owner service role that bypasses real RLS."""

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import aliased

from src import models
from src.config import settings
from src.crud.lifecycle import (
    PrimitiveIdentity,
    PrimitiveKind,
    TenantContext,
    is_primitive_live,
    live_message_row,
    live_peer_row,
    live_session_peer_row,
    live_session_row,
    live_workspace_row,
    primitive_liveness_clause,
)
from src.db import CROSS_TENANT
from src.startup.tenant_isolation_validator import apply_tenant_isolation
from tests.conftest import CONNECTION_URI


@pytest.fixture(scope="session")
def lifecycle_engine(worker_id: str) -> Iterator[Engine]:
    """A separately migrated RLS database with the shared-pool natural keys.

    The migrated dedicated schema keeps a tenant-less compatibility index for
    the previous release's upsert. Shared storage uses the tenant-qualified key.
    Remove that index only in this throwaway fixture, never in the application.
    """
    database = "lifecycle_" + uuid4().hex + "_" + worker_id
    app_role = "lifecycle_app_" + uuid4().hex
    url = CONNECTION_URI.set(database=database)
    maintenance = create_engine(
        CONNECTION_URI.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    with maintenance.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{database}"'))
        conn.execute(text(f"CREATE ROLE {app_role} NOLOGIN NOSUPERUSER NOBYPASSRLS"))
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        root = Path(__file__).resolve().parents[2]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                settings.DB, "CONNECTION_URI", url.render_as_string(hide_password=False)
            )
            patch.setattr(settings.DB, "OWNER_CONNECTION_URI", None)
            config = Config(str(root / "alembic.ini"))
            config.set_main_option("script_location", str(root / "migrations"))
            command.upgrade(config, "head")
        with engine.begin() as conn:
            apply_tenant_isolation(conn, schema="public", app_role=app_role)
            conn.execute(
                text("DROP INDEX public.uq_session_peers_legacy_ws_session_peer")
            )
        yield engine
    finally:
        engine.dispose()
        with maintenance.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
            conn.execute(text(f"DROP ROLE {app_role}"))
        maintenance.dispose()


@pytest.fixture(scope="session")
def lifecycle_service_role(lifecycle_engine: Engine) -> Iterator[str]:
    role = "lifecycle_service_" + uuid4().hex
    with lifecycle_engine.begin() as conn:
        conn.execute(text(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER BYPASSRLS"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))
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
