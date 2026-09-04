"""Flag-on RLS tenant-isolation proof.

The row-level-security policies that enforce tenant isolation are applied to the
shared cloud database out of band, not by this repo, so a plain database has none.
This test applies them itself to a dedicated throwaway database — the way the
alembic harness stands up a real pgvector Postgres — and proves the contract the
per-connection ``app.tenant`` binding relies on: with a tenant bound a session
sees only that tenant's rows, an empty/unset tenant sees none (fail-closed), and a
write for another tenant is refused by the policy's WITH CHECK.

Postgres bypasses row-level security for superusers and table owners, so the
assertions run under ``SET ROLE`` to an ordinary, non-owner role. Connected as the
bootstrap superuser the policies would not apply and the checks would pass without
proving anything.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
)

from src.config import settings
from tests.conftest import CONNECTION_URI

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"

_RLS_TEST_DB_URL: URL = CONNECTION_URI.set(database="rls_isolation_tests")

# The nine tenant-scoped data tables that carry RLS in the cloud deploy.
# The tenants registry and the queue/active_queue_sessions service tables are
# excluded by design — service paths read them across tenants.
_RLS_TABLES: tuple[str, ...] = (
    "workspaces",
    "peers",
    "sessions",
    "messages",
    "message_embeddings",
    "collections",
    "documents",
    "session_peers",
    "webhook_endpoints",
)

# A non-owner, non-superuser role the assertions assume via SET ROLE: RLS is
# bypassed by the owner/superuser, so the checks must run as an ordinary grantee.
_RLS_ROLE = "honcho_rls_probe"


def _new_id() -> str:
    # Workspace.id is a 21-char nanoid (a CHECK enforces its length + alphabet).
    return generate_nanoid()


def _apply_rls_and_role(engine: Engine) -> None:
    """Apply the tenant-isolation policies to the nine data tables, then create the probe role.

    Runs after the schema exists so ``GRANT ... ON ALL TABLES`` covers every table.
    """
    with engine.begin() as conn:
        for table in _RLS_TABLES:
            conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
            conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
            conn.execute(
                text(
                    f"CREATE POLICY tenant_isolation ON {table} "
                    + "USING (tenant_id = current_setting('app.tenant', true)) "
                    + "WITH CHECK (tenant_id = current_setting('app.tenant', true))"
                )
            )

        conn.execute(text(f"DROP ROLE IF EXISTS {_RLS_ROLE}"))
        conn.execute(text(f"CREATE ROLE {_RLS_ROLE} NOLOGIN"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_RLS_ROLE}"))
        conn.execute(
            text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                + f"IN SCHEMA public TO {_RLS_ROLE}"
            )
        )


def _run_on_maintenance_db(*statements: str) -> None:
    """Run AUTOCOMMIT statements against the postgres maintenance database.

    Used for the drops that can't run while connected to the target database (you
    can't drop the database you're in) or inside a transaction block.
    """
    maint = create_engine(
        CONNECTION_URI.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with maint.connect() as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
    finally:
        maint.dispose()


def _force_drop_test_db() -> None:
    # WITH (FORCE) (pg13+) evicts any connection that outlived engine disposal —
    # e.g. the migration engine — which a plain DROP DATABASE would fail on.
    _run_on_maintenance_db(
        f'DROP DATABASE IF EXISTS "{_RLS_TEST_DB_URL.database}" WITH (FORCE)'
    )


def _drop_probe_role() -> None:
    # After the test database (and its grants) are gone, the cluster-global role
    # drops cleanly.
    _run_on_maintenance_db(f"DROP ROLE IF EXISTS {_RLS_ROLE}")


@pytest.fixture(scope="session")
def rls_engine() -> Generator[Engine, None, None]:
    """A dedicated database with the real schema and the cloud RLS policies applied."""
    _force_drop_test_db()  # start clean (e.g. after a leaked prior run)
    create_database(_RLS_TEST_DB_URL)

    # str(URL) masks the password as '***', which then fails auth at migrate; render
    # it with the real password (the workaround the conftest/alembic harness use).
    db_url_str = _RLS_TEST_DB_URL.render_as_string(hide_password=False)
    engine = create_engine(db_url_str)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Build the schema through the real migrations (env.py reads the URL from
        # settings), so it is exactly a self-hoster's schema with the default tenant
        # seeded.
        previous_uri = settings.DB.CONNECTION_URI
        settings.DB.CONNECTION_URI = db_url_str
        try:
            cfg = Config(str(_ALEMBIC_INI))
            cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
            cfg.set_main_option("sqlalchemy.url", db_url_str)
            command.upgrade(cfg, "head")
        finally:
            settings.DB.CONNECTION_URI = previous_uri

        _apply_rls_and_role(engine)
        yield engine
    finally:
        engine.dispose()
        _force_drop_test_db()
        _drop_probe_role()


def _seed_two_tenants(engine: Engine) -> None:
    """Seed two tenants, each with one workspace (as owner, so RLS is bypassed here)."""
    with engine.begin() as conn:
        for tenant in ("tenant-1", "tenant-2"):
            conn.execute(
                text("INSERT INTO tenants (tenant_id) VALUES (:t)"),
                {"t": tenant},
            )
        conn.execute(
            text("INSERT INTO workspaces (tenant_id, id, name) VALUES (:t, :id, :n)"),
            {"t": "tenant-1", "id": _new_id(), "n": "ws-1"},
        )
        conn.execute(
            text("INSERT INTO workspaces (tenant_id, id, name) VALUES (:t, :id, :n)"),
            {"t": "tenant-2", "id": _new_id(), "n": "ws-2"},
        )


def _bind_tenant(conn: Connection, tenant: str) -> None:
    # Session-scoped (is_local=false), mirroring the app's checkout hook — the value
    # persists across statements on this single AUTOCOMMIT connection.
    conn.execute(text("SELECT set_config('app.tenant', :t, false)"), {"t": tenant})


def test_rls_enforces_tenant_isolation(rls_engine: Engine) -> None:
    _seed_two_tenants(rls_engine)

    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        # Become the ordinary grantee so the policies actually apply (a superuser or
        # the table owner would bypass RLS entirely).
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))

        # Bound to tenant-1: only tenant-1's row is visible, even with an explicit
        # cross-tenant predicate.
        _bind_tenant(conn, "tenant-1")
        assert conn.execute(text("SELECT count(*) FROM workspaces")).scalar() == 1
        assert [r[0] for r in conn.execute(text("SELECT name FROM workspaces"))] == [
            "ws-1"
        ]
        assert (
            conn.execute(
                text("SELECT count(*) FROM workspaces WHERE tenant_id = 'tenant-2'")
            ).scalar()
            == 0
        )

        # Fail-closed: an empty/unset tenant matches no rows (current_setting -> '').
        _bind_tenant(conn, "")
        assert conn.execute(text("SELECT count(*) FROM workspaces")).scalar() == 0

        # A cross-tenant write is refused by the policy's WITH CHECK.
        _bind_tenant(conn, "tenant-1")
        with pytest.raises(DBAPIError) as exc:
            conn.execute(
                text(
                    "INSERT INTO workspaces (tenant_id, id, name) "
                    + "VALUES ('tenant-2', :id, 'ws-evil')"
                ),
                {"id": _new_id()},
            )
        assert "row-level security" in str(exc.value).lower()

        # A same-tenant write passes WITH CHECK and is then visible to tenant-1.
        conn.execute(
            text(
                "INSERT INTO workspaces (tenant_id, id, name) "
                + "VALUES ('tenant-1', :id, 'ws-1b')"
            ),
            {"id": _new_id()},
        )
        assert conn.execute(text("SELECT count(*) FROM workspaces")).scalar() == 2

        # Switching the bound tenant flips the visible set — no tenant-1 bleed.
        _bind_tenant(conn, "tenant-2")
        assert [r[0] for r in conn.execute(text("SELECT name FROM workspaces"))] == [
            "ws-2"
        ]

        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()
