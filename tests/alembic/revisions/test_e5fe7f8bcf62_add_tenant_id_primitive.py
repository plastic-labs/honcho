"""Hooks for revision e5fe7f8bcf62 (add_tenant_id_primitive)."""

from __future__ import annotations

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

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
    "document_sources",
    "conclusion_source_messages",
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
        text(f'INSERT INTO "{schema}"."workspaces" ("id", "name") VALUES (:id, :name)'),
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
    _seed_populated_workspace(verifier)
    _PRE_UPGRADE_COUNTS.clear()
    _PRE_UPGRADE_COUNTS.update(_row_counts(verifier, (*_TENANT_SCOPED, "queue")))
    empty = [t for t, n in _PRE_UPGRADE_COUNTS.items() if n == 0]
    assert not empty, f"seed left tables empty: {empty}"


POPULATED_WS = "w_populated"
_PRE_UPGRADE_COUNTS: dict[str, int] = {}


def _row_counts(verifier: MigrationVerifier, tables: tuple[str, ...]) -> dict[str, int]:
    return {
        table: verifier.conn.execute(
            text(f'SELECT COUNT(*) FROM "{verifier.schema}"."{table}"')
        ).scalar_one()
        for table in tables
    }


def _seed_populated_workspace(verifier: MigrationVerifier) -> None:
    """Fill every tenant-scoped table, as a self-hosted c4d8e2f6a1b3 DB would be."""
    conn = verifier.conn
    schema = verifier.schema

    def insert(table: str, row: dict[str, object]) -> None:
        cols = ", ".join(f'"{c}"' for c in row)
        vals = ", ".join(f":{c}" for c in row)
        conn.execute(
            text(f'INSERT INTO "{schema}"."{table}" ({cols}) VALUES ({vals})'), row
        )

    ws = POPULATED_WS
    insert("workspaces", {"id": generate_nanoid(), "name": ws})
    for peer in ("alice", "bob"):
        insert("peers", {"id": generate_nanoid(), "name": peer, "workspace_name": ws})
    session_id = generate_nanoid()
    insert("sessions", {"id": session_id, "name": "chat", "workspace_name": ws})
    for peer in ("alice", "bob"):
        insert(
            "session_peers",
            {"workspace_name": ws, "session_name": "chat", "peer_name": peer},
        )
    message_ids: list[str] = []
    for seq in range(4):
        public_id = generate_nanoid()
        message_ids.append(public_id)
        insert(
            "messages",
            {
                "public_id": public_id,
                "workspace_name": ws,
                "session_name": "chat",
                "peer_name": "alice" if seq % 2 == 0 else "bob",
                "content": f"message {seq}",
                "seq_in_session": seq,
            },
        )
        insert(
            "message_embeddings",
            {
                "content": f"message {seq}",
                "message_id": public_id,
                "workspace_name": ws,
                "session_name": "chat",
                "peer_name": "alice" if seq % 2 == 0 else "bob",
            },
        )
    insert(
        "collections",
        {
            "id": generate_nanoid(),
            "workspace_name": ws,
            "observer": "alice",
            "observed": "alice",
        },
    )
    explicit_id, deductive_id = generate_nanoid(), generate_nanoid()
    for doc_id, level in ((explicit_id, "explicit"), (deductive_id, "deductive")):
        insert(
            "documents",
            {
                "id": doc_id,
                "workspace_name": ws,
                "observer": "alice",
                "observed": "alice",
                "content": f"{level} conclusion",
                "level": level,
                "session_name": "chat",
            },
        )
    insert(
        "document_sources",
        {"derived_id": deductive_id, "source_id": explicit_id, "workspace_name": ws},
    )
    for position, message_id in enumerate(message_ids[:2]):
        insert(
            "conclusion_source_messages",
            {
                "conclusion_id": explicit_id,
                "message_id": message_id,
                "position": position,
                "workspace_name": ws,
            },
        )
    insert(
        "webhook_endpoints",
        {"id": generate_nanoid(), "workspace_name": ws, "url": "http://localhost/hook"},
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("session_id", "payload", "task_type", "work_unit_key", "workspace_name") '
            + "VALUES (:sid, CAST(:payload AS jsonb), 'representation', :key, :ws)"
        ),
        {
            "sid": session_id,
            "payload": '{"message_ids": []}',
            "key": f"representation:{ws}:chat:alice",
            "ws": ws,
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
        text(f'SELECT COUNT(*) FROM "{schema}"."tenants" WHERE tenant_id = \'default\'')
    ).scalar()
    assert default_count == 1, f"expected the default tenant, found {default_count}"

    # vector_correlation_id is unique over set keys; NULLs may repeat.
    verifier.assert_indexes_exist([("tenants", "uq_tenants_vector_correlation_id")])
    verifier.assert_indexes_not_exist([("tenants", "ix_tenants_vector_correlation_id")])
    insert_tenant = text(
        f'INSERT INTO "{schema}"."tenants" ("tenant_id", "tier", "vector_correlation_id") '
        + "VALUES (:tid, 'shared', :key)"
    )
    with verifier.conn.begin_nested():
        conn.execute(insert_tenant, {"tid": "t-vcid-a", "key": "vec-a"})
        conn.execute(insert_tenant, {"tid": "t-vcid-null-1", "key": None})
        conn.execute(insert_tenant, {"tid": "t-vcid-null-2", "key": None})
    with pytest.raises(IntegrityError) as rejection, verifier.conn.begin_nested():
        conn.execute(insert_tenant, {"tid": "t-vcid-dup", "key": "vec-a"})
    assert "uq_tenants_vector_correlation_id" in str(rejection.value)

    # tenant_id is NOT NULL on every tenant-scoped table, and existing rows backfilled.
    for table in _TENANT_SCOPED:
        verifier.assert_column_exists(table, "tenant_id", nullable=False)
    after = _row_counts(verifier, (*_TENANT_SCOPED, "queue"))
    assert after == _PRE_UPGRADE_COUNTS, (
        f"row counts changed across the upgrade: {_PRE_UPGRADE_COUNTS} -> {after}"
    )
    queue_tenants = conn.execute(
        text(f'SELECT COUNT(tenant_id) FROM "{schema}"."queue"')
    ).scalar_one()
    assert queue_tenants == 0, "queue rows must stay tenant-less (NULL) flag-off"
    for table in _TENANT_SCOPED:
        verifier.assert_no_nulls(table, "tenant_id")
        backfilled = conn.execute(
            text(
                f'SELECT COUNT(*) FROM "{schema}"."{table}" '
                + "WHERE tenant_id <> 'default'"
            )
        ).scalar()
        assert backfilled == 0, f"{table} has rows not backfilled to the default tenant"

    # The previous release never sets tenant_id; its inserts get the instance tenant.
    with verifier.conn.begin_nested() as old_writer:
        tenant = conn.execute(
            text(
                f'INSERT INTO "{schema}"."workspaces" ("id", "name") '
                + "VALUES (:id, :name) RETURNING tenant_id"
            ),
            {"id": generate_nanoid(), "name": "written-by-old-image"},
        ).scalar()
        old_writer.rollback()
    assert tenant == "default", f"old-image insert got tenant {tenant!r}"

    # Composite PK (tenant_id, id) on workspaces.
    pk = verifier.get_inspector().get_pk_constraint("workspaces", schema=schema)
    assert pk["constrained_columns"] == [
        "tenant_id",
        "id",
    ], f"workspaces PK is {pk['constrained_columns']}"

    # document_sources arrives from a7c3e9f1b2d4 keyed on documents.id alone; it
    # must leave keyed through the tenant, or its edges cannot reach a document
    # whose key is now (tenant_id, id).
    pk = verifier.get_inspector().get_pk_constraint("document_sources", schema=schema)
    assert pk["constrained_columns"] == [
        "tenant_id",
        "derived_id",
        "source_id",
    ], f"document_sources PK is {pk['constrained_columns']}"
    verifier.assert_constraint_exists(
        "document_sources",
        "fk_document_sources_derived_tenant_documents",
        "foreign_key",
    )

    # conclusion_source_messages arrives from c4d8e2f6a1b3 keyed on documents.id
    # and messages.public_id alone, and leaves keyed through the tenant too.
    pk = verifier.get_inspector().get_pk_constraint(
        "conclusion_source_messages", schema=schema
    )
    assert pk["constrained_columns"] == [
        "tenant_id",
        "conclusion_id",
        "message_id",
    ], f"conclusion_source_messages PK is {pk['constrained_columns']}"
    verifier.assert_constraint_exists(
        "conclusion_source_messages",
        "fk_conclusion_source_messages_conclusion_tenant_documents",
        "foreign_key",
    )
    verifier.assert_constraint_exists(
        "conclusion_source_messages",
        "fk_conclusion_source_messages_message_tenant_messages",
        "foreign_key",
    )

    # Tenant-scoped uniqueness replaced the global one.
    verifier.assert_constraint_exists(
        "workspaces", "uq_workspaces_tenant_id_name", "unique"
    )

    # The previous release's session-peer upsert still finds its conflict target.
    verifier.assert_indexes_exist(
        [("session_peers", "uq_session_peers_legacy_ws_session_peer")]
    )
    legacy_upsert = text(
        f'INSERT INTO "{schema}"."session_peers" '
        + "(workspace_name, session_name, peer_name) VALUES (:ws, :sn, :pn) "
        + "ON CONFLICT (session_name, peer_name, workspace_name) "
        + "DO UPDATE SET left_at = NULL"
    )
    upsert_params = {"ws": WORKSPACE_NAME, "sn": SESSION_NAME, "pn": PEER_NAME}
    with verifier.conn.begin_nested():
        conn.execute(legacy_upsert, upsert_params)
        conn.execute(legacy_upsert, upsert_params)

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
