"""Hooks for revision e5fe7f8bcf62 (add_tenant_id_primitive)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = "w1"
PEER_NAME = "p1"
SESSION_NAME = "s1"

# Tenant-scoped tables that must end up with a NOT NULL tenant_id.
_TENANT_SCOPED = (
    "workspaces",
    "peers",
    "sessions",
    "messages",
    "message_embeddings",
    "collections",
    "documents",
    "webhook_endpoints",
    "session_peers",
)


@register_before_upgrade("e5fe7f8bcf62")
def prepare_add_tenant_id_primitive(verifier: MigrationVerifier) -> None:
    """Seed old-schema rows before upgrading to e5fe7f8bcf62."""
    # Pre-state: no tenants table, no tenant_id yet.
    verifier.assert_table_exists("tenants", exists=False)
    verifier.assert_column_exists("workspaces", "tenant_id", exists=False)
    verifier.assert_column_exists("messages", "tenant_id", exists=False)

    conn = verifier.conn
    schema = verifier.schema

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "name") VALUES (:id, :name)'
        ),
        {"id": generate_nanoid(), "name": WORKSPACE_NAME},
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
            + "VALUES (:id, :name, :ws)"
        ),
        {"id": generate_nanoid(), "name": PEER_NAME, "ws": WORKSPACE_NAME},
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name") '
            + "VALUES (:id, :name, :ws)"
        ),
        {"id": generate_nanoid(), "name": SESSION_NAME, "ws": WORKSPACE_NAME},
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."messages" '
            + '("public_id", "workspace_name", "session_name", "peer_name", "content", "seq_in_session") '
            + "VALUES (:pid, :ws, :sn, :pn, :content, :seq)"
        ),
        {
            "pid": generate_nanoid(),
            "ws": WORKSPACE_NAME,
            "sn": SESSION_NAME,
            "pn": PEER_NAME,
            "content": "hello",
            "seq": 0,
        },
    )


@register_after_upgrade("e5fe7f8bcf62")
def verify_add_tenant_id_primitive(verifier: MigrationVerifier) -> None:
    """Assert the tenant_id primitive landed."""
    schema = verifier.schema
    conn = verifier.conn

    # tenants table + the renamed correlation column + the default tenant row.
    verifier.assert_table_exists("tenants")
    verifier.assert_column_exists("tenants", "vector_correlation_id")
    verifier.assert_column_exists("tenants", "legacy_app_name", exists=False)
    default_count = conn.execute(
        text(f"SELECT COUNT(*) FROM \"{schema}\".\"tenants\" WHERE tenant_id = 'default'")
    ).scalar()
    assert default_count == 1, f"expected the default tenant, found {default_count}"

    # tenant_id is NOT NULL on every tenant-scoped table, and existing rows backfilled.
    for table in _TENANT_SCOPED:
        verifier.assert_column_exists(table, "tenant_id", nullable=False)
    for table in ("workspaces", "peers", "sessions", "messages"):
        verifier.assert_no_nulls(table, "tenant_id")
        backfilled = conn.execute(
            text(
                f'SELECT COUNT(*) FROM "{schema}"."{table}" '
                + "WHERE tenant_id <> 'default'"
            )
        ).scalar()
        assert backfilled == 0, f"{table} has rows not backfilled to the default tenant"

    # Composite PK (tenant_id, id) on workspaces.
    pk = verifier.get_inspector().get_pk_constraint("workspaces", schema=schema)
    assert pk["constrained_columns"] == [
        "tenant_id",
        "id",
    ], f"workspaces PK is {pk['constrained_columns']}"

    # Tenant-scoped uniqueness replaced the global one.
    verifier.assert_constraint_exists(
        "workspaces", "uq_workspaces_tenant_id_name", "unique"
    )

    # tenant_id FK to tenants + a representative composite FK.
    verifier.assert_constraint_exists(
        "peers", "fk_peers_tenant_id_tenants", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "peers", "fk_peers_ws_tenant_workspaces", "foreign_key"
    )

    # Service tables carry tenant_id but leave it nullable (attribution only).
    verifier.assert_column_exists("queue", "tenant_id", nullable=True)
    verifier.assert_column_exists("active_queue_sessions", "tenant_id", nullable=True)
