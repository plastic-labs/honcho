"""Flag-on RLS tenant-isolation proof.

The row-level-security policies that enforce tenant isolation are applied by
``apply_tenant_isolation`` after the migrations run under ``MULTI_TENANT``, so a
plain single-tenant database has none. This test applies them the same way to a
dedicated throwaway database — the way the alembic harness stands up a real
pgvector Postgres — and proves the contract the
per-connection ``app.tenant`` binding relies on: with a tenant bound a session
sees only that tenant's rows, an empty/unset tenant sees none (fail-closed), and a
write for another tenant is refused by the policy's WITH CHECK.

Postgres bypasses row-level security for superusers and table owners, so the
assertions run under ``SET ROLE`` to an ordinary, non-owner role. Connected as the
bootstrap superuser the policies would not apply and the checks would pass without
proving anything.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from src.models import MESSAGE_TEXT_MATCHES, MESSAGE_TEXT_MATCHES_ARGS
from src.startup.tenant_isolation_validator import (
    _RLS_REQUIRED_TABLES,  # pyright: ignore[reportPrivateUsage]
    apply_tenant_isolation,
)
from tests.conftest import (
    CONNECTION_URI,
    _get_test_db_url,  # pyright: ignore[reportPrivateUsage]
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"

# Per xdist worker: the database and the two roles are cluster-global names, so
# two workers running this module would otherwise create and drop each other's.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "master")
_RLS_TEST_DB_URL: URL = _get_test_db_url(_WORKER, tag="rls")

# A non-owner, non-superuser role the assertions assume via SET ROLE: RLS is
# bypassed by the owner/superuser, so the checks must run as an ordinary grantee.
_RLS_ROLE = f"honcho_rls_probe_{_WORKER}"

# A role with USAGE on the schema and nothing else: it must not be able to call
# the SECURITY DEFINER search function.
_STRANGER_ROLE = f"honcho_rls_stranger_{_WORKER}"


def _new_id() -> str:
    # Workspace.id is a 21-char nanoid (a CHECK enforces its length + alphabet).
    return generate_nanoid()


def _apply_rls_and_role(engine: Engine) -> None:
    """Create the probe role, then apply the tenant-isolation policies and grants to it.

    Runs after the schema exists so ``GRANT ... ON ALL TABLES`` covers every table.
    """
    with engine.begin() as conn:
        conn.execute(text(f"DROP ROLE IF EXISTS {_RLS_ROLE}"))
        conn.execute(text(f"CREATE ROLE {_RLS_ROLE} NOLOGIN"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_RLS_ROLE}"))
        apply_tenant_isolation(conn, schema="public", app_role=_RLS_ROLE)
        conn.execute(
            text(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                + f"IN SCHEMA public TO {_RLS_ROLE}"
            )
        )
        conn.execute(text(f"DROP ROLE IF EXISTS {_STRANGER_ROLE}"))
        conn.execute(text(f"CREATE ROLE {_STRANGER_ROLE} NOLOGIN"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_STRANGER_ROLE}"))


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
    _run_on_maintenance_db(
        f"DROP ROLE IF EXISTS {_RLS_ROLE}", f"DROP ROLE IF EXISTS {_STRANGER_ROLE}"
    )


@pytest.fixture(scope="session")
def rls_engine() -> Generator[Engine]:
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
        previous_owner_uri = settings.DB.OWNER_CONNECTION_URI
        settings.DB.CONNECTION_URI = db_url_str
        settings.DB.OWNER_CONNECTION_URI = None
        try:
            cfg = Config(str(_ALEMBIC_INI))
            cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
            cfg.set_main_option("sqlalchemy.url", db_url_str)
            command.upgrade(cfg, "head")
        finally:
            settings.DB.CONNECTION_URI = previous_uri
            settings.DB.OWNER_CONNECTION_URI = previous_owner_uri

        _apply_rls_and_role(engine)
        # Seed here, not in the first test that needs it: under xdist the tests of
        # this module land on different workers, each with its own database.
        _seed_two_tenants(engine)
        _seed_same_named_workspaces(engine)
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


def test_rls_predicate_preserves_index_use(rls_engine: Engine) -> None:
    # The RLS USING qual (tenant_id = current_setting('app.tenant', true)) must not
    # defeat the tenant-leading indexes: for a tenant-scoped lookup the planner
    # should still reach an index — the qual stays sargable — rather than fall back
    # to a sequential scan + post-filter. Seq scans are penalized here so the plan
    # reflects the qual's effect, not table size.
    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))
        _bind_tenant(conn, "tenant-1")
        conn.execute(text("SET enable_seqscan = off"))
        plan = "\n".join(
            str(row[0])
            for row in conn.execute(
                text("EXPLAIN SELECT id FROM peers WHERE workspace_name = 'ws-1'")
            )
        )
        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()

    assert "Index Scan" in plan or "Index Only Scan" in plan, plan
    assert "Seq Scan" not in plan, plan


def _seed_same_named_workspaces(engine: Engine) -> None:
    """Two tenants with a same-named workspace and same-content messages."""
    with engine.begin() as conn:
        conn.execute(text("RESET ROLE"))
        for n, tenant in enumerate(("t-search-1", "t-search-2"), start=1):
            conn.execute(
                text("INSERT INTO tenants (tenant_id) VALUES (:t)"), {"t": tenant}
            )
            conn.execute(
                text(
                    "INSERT INTO workspaces (tenant_id, id, name) VALUES (:t, :id, :n)"
                ),
                {"t": tenant, "id": _new_id(), "n": "ws-search"},
            )
            conn.execute(
                text(
                    "INSERT INTO peers (tenant_id, id, name, workspace_name)"
                    + " VALUES (:t, :id, 'p', 'ws-search')"
                ),
                {"t": tenant, "id": _new_id()},
            )
            conn.execute(
                text(
                    "INSERT INTO sessions (tenant_id, id, name, workspace_name)"
                    + " VALUES (:t, :id, 's', 'ws-search')"
                ),
                {"t": tenant, "id": _new_id()},
            )
            for seq, content in ((1, "the quick brown fox"), (2, "nothing here")):
                conn.execute(
                    text(
                        "INSERT INTO messages (tenant_id, id, public_id, workspace_name,"
                        + " session_name, peer_name, content, seq_in_session,"
                        + " token_count) VALUES (:t, :id, :pid, 'ws-search', 's', 'p',"
                        + " :c, :seq, 4)"
                    ),
                    {
                        "t": tenant,
                        "id": n * 100 + seq,
                        "pid": _new_id(),
                        "c": content,
                        "seq": seq,
                    },
                )
        conn.execute(
            text(
                "INSERT INTO messages (tenant_id, id, public_id, workspace_name,"
                + " session_name, peer_name, content, seq_in_session, token_count)"
                + " SELECT 't-search-1', 1000 + g, 'a1e7ce5396d5' || lpad(g::text, 9, '0'),"
                + " 'ws-search', 's', 'p', 'filler message number ' || g, 10 + g, 4"
                + " FROM generate_series(1, 3000) AS g"
            )
        )
        conn.execute(text("ANALYZE messages"))


def _matches(
    conn: Connection,
    tenant: str,
    term: str = "fox",
    *,
    after: tuple[int, object, float] | None = None,
    limit: int = 50,
) -> list[tuple[int, object, float]]:
    id_, created_at, rank = after if after is not None else (None, None, None)
    rows = conn.execute(
        text(
            f"SELECT * FROM {MESSAGE_TEXT_MATCHES}(:t, 'ws-search', :q, :p, true,"
            + " CAST(:rank AS double precision), CAST(:created_at AS timestamptz),"
            + " CAST(:id AS bigint), :limit)"
        ),
        {
            "t": tenant,
            "q": term,
            "p": f"%{term}%",
            "rank": rank,
            "created_at": created_at,
            "id": id_,
            "limit": limit,
        },
    )
    return [(int(r[0]), r[1], float(r[2])) for r in rows]


def _ids(rows: list[tuple[int, object, float]]) -> list[int]:
    return [row[0] for row in rows]


def test_search_function_reads_only_the_bound_tenant(rls_engine: Engine) -> None:

    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))

        _bind_tenant(conn, "t-search-1")
        assert _ids(_matches(conn, "t-search-1")) == [101]
        assert _ids(_matches(conn, "t-search-2")) == []

        _bind_tenant(conn, "t-search-2")
        assert _ids(_matches(conn, "t-search-2")) == [201]
        assert _ids(_matches(conn, "t-search-1")) == []

        _bind_tenant(conn, "")
        assert _ids(_matches(conn, "t-search-1")) == []
        assert _ids(_matches(conn, "")) == []

        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()


def test_search_function_pages_newest_first(rls_engine: Engine) -> None:
    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))
        _bind_tenant(conn, "t-search-1")

        first = _matches(conn, "t-search-1", "filler message", limit=10)
        second = _matches(
            conn, "t-search-1", "filler message", after=first[-1], limit=10
        )
        everything = _matches(conn, "t-search-1", "filler message", limit=5000)

        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()

    assert len(first) == 10 and len(second) == 10
    assert _ids(first) == list(range(4000, 3990, -1))
    assert _ids(second) == list(range(3990, 3980, -1))
    assert len(everything) == 3000
    assert _ids(everything) == sorted(_ids(everything), reverse=True)


def test_search_function_is_not_executable_by_an_ungranted_role(
    rls_engine: Engine,
) -> None:
    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text(f"SET ROLE {_STRANGER_ROLE}"))
        _bind_tenant(conn, "t-search-1")
        with pytest.raises(DBAPIError) as exc:
            _matches(conn, "t-search-1")
        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()
    assert "permission denied for function" in str(exc.value)


_FTS_MATCH = "to_tsvector('english', content) @@ plainto_tsquery('english', 'fox')"
_ILIKE_MATCH = "content ILIKE '%fox%'"


def _search_plan(conn: Connection, match: str = _FTS_MATCH) -> str:
    conn.execute(text("SET enable_seqscan = off"))
    rows = conn.execute(
        text(
            "EXPLAIN SELECT id FROM messages WHERE tenant_id = 't-search-1'"
            + " AND workspace_name = 'ws-search'"
            + f" AND {match}"
        )
    )
    return "\n".join(str(r[0]) for r in rows)


def test_rls_keeps_the_content_index_from_the_app_role(rls_engine: Engine) -> None:
    # The function exists because @@ is not leakproof: for the app role the policy
    # qual runs first and the GIN index cannot serve the match. The owner, who
    # bypasses RLS, gets the index. The function runs as the owner.
    raw = rls_engine.connect()
    conn = raw.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(text("RESET ROLE"))
        _bind_tenant(conn, "t-search-1")
        owner_plan = _search_plan(conn)
        conn.execute(text(f"SET ROLE {_RLS_ROLE}"))
        app_plan = _search_plan(conn)
        conn.execute(text("RESET ROLE"))
    finally:
        raw.close()

    assert "ix_messages_tenant_content_fts" in owner_plan, owner_plan
    assert "ix_messages_tenant_content_fts" not in app_plan, app_plan


def test_text_indexes_lead_with_the_tenant_and_workspace(rls_engine: Engine) -> None:
    # Both text matches reach a GIN index whose first keys are tenant_id and
    # workspace_name, so the bitmap is already narrowed to the workspace, and the
    # tenant and workspace predicates are index conditions, not a recheck filter.
    # The workspace is padded to 40k rows inside a rolled-back transaction: with
    # 3k rows the planner prefers a btree on workspace_name plus an ILIKE filter.
    with rls_engine.connect() as conn:
        seeded = conn.scalar(
            text("SELECT 1 FROM tenants WHERE tenant_id = 't-search-1'")
        )
    if not seeded:
        _seed_same_named_workspaces(rls_engine)

    with rls_engine.connect() as conn:
        conn.execute(text("RESET ROLE"))
        _bind_tenant(conn, "t-search-1")
        conn.execute(
            text(
                "INSERT INTO messages (tenant_id, id, public_id, workspace_name,"
                + " session_name, peer_name, content, seq_in_session, token_count)"
                + " SELECT 't-search-1', 100000 + g, 'd4b8e2f7a913' || lpad(g::text, 9, '0'),"
                + " 'ws-search', 's', 'p', 'padding message number ' || g, 50000 + g, 4"
                + " FROM generate_series(1, 40000) AS g"
            )
        )
        conn.execute(text("ANALYZE messages"))
        fts_plan = _search_plan(conn, _FTS_MATCH)
        ilike_plan = _search_plan(conn, _ILIKE_MATCH)
        conn.rollback()

    for plan, index in (
        (fts_plan, "ix_messages_tenant_content_fts"),
        (ilike_plan, "ix_messages_tenant_content_trgm"),
    ):
        assert f"Bitmap Index Scan on {index}" in plan, plan
        index_cond = next(line for line in plan.splitlines() if "Index Cond" in line)
        assert "tenant_id" in index_cond and "workspace_name" in index_cond, plan


def test_apply_tenant_isolation_is_idempotent(rls_engine: Engine) -> None:
    with rls_engine.begin() as conn:
        apply_tenant_isolation(conn, schema="public", app_role=_RLS_ROLE)
        rows = conn.execute(
            text(
                "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,"
                + " (SELECT count(*) FROM pg_policy p"
                + "  WHERE p.polrelid = c.oid AND p.polname = 'tenant_isolation'),"
                + " has_table_privilege(:role, c.oid, 'SELECT, INSERT, UPDATE, DELETE')"
                + " FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
                + " WHERE n.nspname = 'public' AND c.relname = ANY(:tables)"
            ),
            {"role": _RLS_ROLE, "tables": list(_RLS_REQUIRED_TABLES)},
        ).all()
    assert {row[0] for row in rows} == set(_RLS_REQUIRED_TABLES)
    assert all(row[1] and row[2] and row[3] == 1 and row[4] for row in rows)


def test_upgrade_restores_isolation_under_multi_tenant(rls_engine: Engine) -> None:
    with rls_engine.begin() as conn:
        conn.execute(text("DROP POLICY tenant_isolation ON webhook_endpoints"))
        conn.execute(text("ALTER TABLE webhook_endpoints DISABLE ROW LEVEL SECURITY"))
        conn.execute(text("ALTER TABLE webhook_endpoints NO FORCE ROW LEVEL SECURITY"))
        conn.execute(text(f"REVOKE ALL ON webhook_endpoints FROM {_RLS_ROLE}"))
        conn.execute(
            text(
                f"REVOKE ALL ON FUNCTION {MESSAGE_TEXT_MATCHES}"
                + f"{MESSAGE_TEXT_MATCHES_ARGS} FROM {_RLS_ROLE}"
            )
        )

    owner_url = _RLS_TEST_DB_URL.render_as_string(hide_password=False)
    app_url = _RLS_TEST_DB_URL.set(username=_RLS_ROLE).render_as_string(
        hide_password=False
    )
    previous = (
        settings.DB.CONNECTION_URI,
        settings.DB.OWNER_CONNECTION_URI,
        settings.DB.SERVICE_CONNECTION_URI,
        settings.MULTI_TENANT,
    )
    settings.DB.CONNECTION_URI = app_url
    settings.DB.OWNER_CONNECTION_URI = owner_url
    settings.DB.SERVICE_CONNECTION_URI = None
    settings.MULTI_TENANT = True
    try:
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        command.upgrade(cfg, "head")
    finally:
        (
            settings.DB.CONNECTION_URI,
            settings.DB.OWNER_CONNECTION_URI,
            settings.DB.SERVICE_CONNECTION_URI,
            settings.MULTI_TENANT,
        ) = previous

    with rls_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT c.relrowsecurity, c.relforcerowsecurity,"
                + " (SELECT count(*) FROM pg_policy p WHERE p.polrelid = c.oid),"
                + " has_table_privilege(:role, c.oid, 'SELECT, INSERT, UPDATE, DELETE'),"
                + " has_function_privilege(:role, CAST(:function AS regprocedure),"
                + " 'EXECUTE')"
                + " FROM pg_class c WHERE c.oid = 'webhook_endpoints'::regclass"
            ),
            {
                "role": _RLS_ROLE,
                "function": f"{MESSAGE_TEXT_MATCHES}{MESSAGE_TEXT_MATCHES_ARGS}",
            },
        ).one()
    assert tuple(row) == (True, True, 1, True, True)


@contextmanager
def _migrating_as_owner(*, multi_tenant: bool) -> Generator[Config]:
    """Settings for running alembic as the database owner with the probe as app role."""
    previous = (
        settings.DB.CONNECTION_URI,
        settings.DB.OWNER_CONNECTION_URI,
        settings.DB.SERVICE_CONNECTION_URI,
        settings.MULTI_TENANT,
    )
    settings.DB.CONNECTION_URI = _RLS_TEST_DB_URL.set(
        username=_RLS_ROLE
    ).render_as_string(hide_password=False)
    settings.DB.OWNER_CONNECTION_URI = _RLS_TEST_DB_URL.render_as_string(
        hide_password=False
    )
    settings.DB.SERVICE_CONNECTION_URI = None
    settings.MULTI_TENANT = multi_tenant
    try:
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        yield cfg
    finally:
        (
            settings.DB.CONNECTION_URI,
            settings.DB.OWNER_CONNECTION_URI,
            settings.DB.SERVICE_CONNECTION_URI,
            settings.MULTI_TENANT,
        ) = previous


def test_flag_off_upgrade_grants_the_app_role(rls_engine: Engine) -> None:
    """A separate owner creates the tables flag-off too; the app role still gets them."""
    with rls_engine.begin() as conn:
        conn.execute(text(f"REVOKE ALL ON queue FROM {_RLS_ROLE}"))

    with _migrating_as_owner(multi_tenant=False) as cfg:
        command.upgrade(cfg, "head")

    with rls_engine.connect() as conn:
        granted = conn.execute(
            text(
                "SELECT has_table_privilege(:role, 'queue',"
                + " 'SELECT, INSERT, UPDATE, DELETE')"
            ),
            {"role": _RLS_ROLE},
        ).scalar_one()
    assert granted


def test_isolation_is_applied_only_at_head(rls_engine: Engine) -> None:
    """Below head the policy's columns or the search function may not exist yet."""
    with rls_engine.begin() as conn:
        conn.execute(text("DROP POLICY tenant_isolation ON webhook_endpoints"))

    def policies() -> int:
        with rls_engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT count(*) FROM pg_policy"
                    + " WHERE polrelid = 'webhook_endpoints'::regclass"
                )
            ).scalar_one()

    with _migrating_as_owner(multi_tenant=True) as cfg:
        try:
            command.downgrade(cfg, "-1")
            assert policies() == 0
        finally:
            command.upgrade(cfg, "head")
    assert policies() == 1


@pytest.mark.asyncio
async def test_partition_check_sees_write_only_privileges(
    rls_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role that can DELETE from a partition, but not SELECT, still bypasses RLS."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.startup import tenant_isolation_validator as validator

    with rls_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS probe_parent"))
        conn.execute(
            text(
                "CREATE TABLE probe_parent (tenant_id text, id int)"
                + " PARTITION BY HASH (tenant_id)"
            )
        )
        for remainder in (0, 1):
            conn.execute(
                text(
                    f"CREATE TABLE probe_parent_p{remainder} PARTITION OF probe_parent"
                    + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                )
            )
        conn.execute(text(f"GRANT DELETE ON probe_parent_p1 TO {_RLS_ROLE}"))
    monkeypatch.setattr(validator, "_RLS_REQUIRED_TABLES", ("probe_parent",))
    engine = create_async_engine(_RLS_TEST_DB_URL.render_as_string(hide_password=False))

    @event.listens_for(engine.sync_engine, "connect")
    def _as_probe_role(dbapi_connection: Any, _record: Any) -> None:  # pyright: ignore[reportUnusedFunction]
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET ROLE {_RLS_ROLE}")
        cursor.close()

    try:
        accessible = await validator._introspect_accessible_partitions_once(  # pyright: ignore[reportPrivateUsage]
            engine, "public"
        )
    finally:
        await engine.dispose()
        with rls_engine.begin() as conn:
            conn.execute(text("DROP TABLE probe_parent"))
    assert accessible == ["probe_parent_p1"]
