"""add tenant id primitive

Revision ID: e5fe7f8bcf62
Revises: e4eba9cfaa6f
Create Date: 2026-09-03

Adds ``tenant_id`` as a first-class primitive to the data model: a new
``tenants`` table and a ``tenant_id`` column (plus tenant-scoped composite
PKs / uniques / FKs / indexes) on every tenant-scoped table, matching the
declarative models MINUS physical partitioning.
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    column_exists,
    get_schema,
    index_exists,
    table_exists,
)

# revision identifiers, used by Alembic.
revision: str = "e5fe7f8bcf62"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

# The single tenant every existing row is backfilled to on a self-host install.
DEFAULT_TENANT_ID = "default"

# Tables that gain tenant_id + a composite PK + composite FKs (session_peers has
# an all-natural composite PK and is handled with the rest).
TENANT_SCOPED: tuple[str, ...] = (
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

# New primary keys (table -> ordered PK columns). Everything is (tenant_id, id)
# except the association table, whose PK is all-natural.
NEW_PKS: dict[str, list[str]] = {
    "workspaces": ["tenant_id", "id"],
    "peers": ["tenant_id", "id"],
    "sessions": ["tenant_id", "id"],
    "messages": ["tenant_id", "id"],
    "message_embeddings": ["tenant_id", "id"],
    "collections": ["tenant_id", "id"],
    "documents": ["tenant_id", "id"],
    "webhook_endpoints": ["tenant_id", "id"],
    "session_peers": ["tenant_id", "workspace_name", "session_name", "peer_name"],
}

# New unique constraints (table -> list of (name, cols)).
NEW_UNIQUES: dict[str, list[tuple[str, list[str]]]] = {
    "workspaces": [("uq_workspaces_tenant_id_name", ["tenant_id", "name"])],
    "peers": [
        ("uq_peers_tenant_id_name_workspace_name", ["tenant_id", "name", "workspace_name"])
    ],
    "sessions": [
        ("uq_sessions_tenant_id_name_workspace_name", ["tenant_id", "name", "workspace_name"])
    ],
    "messages": [
        ("uq_messages_tenant_id_public_id", ["tenant_id", "public_id"]),
        (
            "uq_messages_tenant_id_ws_session_seq",
            ["tenant_id", "workspace_name", "session_name", "seq_in_session"],
        ),
    ],
    "collections": [
        (
            "uq_collections_tenant_id_observer_observed_ws",
            ["tenant_id", "observer", "observed", "workspace_name"],
        )
    ],
}

# New foreign keys: (name, source_table, [local_cols], ref_table, [ref_cols], ondelete)
# Every tenant-scoped table also FKs its tenant_id -> tenants.tenant_id (added in a
# loop below); the composite FKs below carry the tenant_id into the natural-key refs.
NEW_FKS: list[tuple[str, str, list[str], str, list[str], str | None]] = [
    # peers
    ("fk_peers_ws_tenant_workspaces", "peers", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    # sessions
    ("fk_sessions_ws_tenant_workspaces", "sessions", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    # messages
    ("fk_messages_session_ws_tenant_sessions", "messages", ["session_name", "workspace_name", "tenant_id"], "sessions", ["name", "workspace_name", "tenant_id"], None),
    ("fk_messages_peer_ws_tenant_peers", "messages", ["peer_name", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    # message_embeddings
    ("fk_msg_emb_tenant_message_messages", "message_embeddings", ["tenant_id", "message_id"], "messages", ["tenant_id", "public_id"], "CASCADE"),
    ("fk_msg_emb_ws_tenant_workspaces", "message_embeddings", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    ("fk_msg_emb_session_ws_tenant_sessions", "message_embeddings", ["session_name", "workspace_name", "tenant_id"], "sessions", ["name", "workspace_name", "tenant_id"], None),
    ("fk_msg_emb_peer_ws_tenant_peers", "message_embeddings", ["peer_name", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    # collections
    ("fk_collections_ws_tenant_workspaces", "collections", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    ("fk_collections_observer_ws_tenant_peers", "collections", ["observer", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    ("fk_collections_observed_ws_tenant_peers", "collections", ["observed", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    # documents
    ("fk_documents_ws_tenant_workspaces", "documents", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    ("fk_documents_collection_tenant_collections", "documents", ["observer", "observed", "workspace_name", "tenant_id"], "collections", ["observer", "observed", "workspace_name", "tenant_id"], None),
    ("fk_documents_observer_ws_tenant_peers", "documents", ["observer", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    ("fk_documents_observed_ws_tenant_peers", "documents", ["observed", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
    ("fk_documents_session_ws_tenant_sessions", "documents", ["session_name", "workspace_name", "tenant_id"], "sessions", ["name", "workspace_name", "tenant_id"], None),
    # webhook_endpoints
    ("fk_webhook_ws_tenant_workspaces", "webhook_endpoints", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    # session_peers
    ("fk_session_peers_ws_tenant_workspaces", "session_peers", ["workspace_name", "tenant_id"], "workspaces", ["name", "tenant_id"], None),
    ("fk_session_peers_session_ws_tenant_sessions", "session_peers", ["session_name", "workspace_name", "tenant_id"], "sessions", ["name", "workspace_name", "tenant_id"], None),
    ("fk_session_peers_peer_ws_tenant_peers", "session_peers", ["peer_name", "workspace_name", "tenant_id"], "peers", ["name", "workspace_name", "tenant_id"], None),
]


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _drop_all_fks(tables: Sequence[str]) -> None:
    """Drop every FK on the given tables (dynamic — old names are unknown)."""
    insp = _inspector()
    for tname in tables:
        if not table_exists(tname):
            continue
        for fk in insp.get_foreign_keys(tname, schema=schema):
            name = fk.get("name")
            if name:
                op.drop_constraint(name, tname, type_="foreignkey", schema=schema)


def _drop_pk(table: str) -> None:
    insp = _inspector()
    pk = insp.get_pk_constraint(table, schema=schema)
    name = pk.get("name") if pk else None
    if name:
        op.drop_constraint(name, table, type_="primary", schema=schema)


def _drop_all_uniques(table: str) -> None:
    """Drop unique constraints and pure unique indexes (all become tenant-scoped)."""
    insp = _inspector()
    for uq in insp.get_unique_constraints(table, schema=schema):
        name = uq.get("name")
        if name:
            op.drop_constraint(name, table, type_="unique", schema=schema)
    for idx in insp.get_indexes(table, schema=schema):
        idx_name = idx.get("name")
        if idx.get("unique") and idx_name:
            op.drop_index(idx_name, table_name=table, schema=schema)


def upgrade() -> None:
    # region ai
    # Disable-on-prod + idempotency guard. Prod's shared schema is built by the
    # bootstrap (which creates `tenants`) and prod is `alembic stamp`-ed past this
    # revision, so it never runs the body; this also makes the migration a no-op on
    # any DB that already has the tenant schema (re-runs). Self-host / prosumer DBs
    # have no `tenants` table at this point, so they run the full transform.
    # endregion
    if table_exists("tenants"):
        return

    # 1. tenants table + its index + the single self-host tenant.
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.TEXT(), nullable=False),
        sa.Column("vector_correlation_id", sa.TEXT(), nullable=True),
        sa.Column("tier", sa.TEXT(), server_default="dedicated", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("tenant_id", name="pk_tenants"),
        schema=schema,
    )
    op.create_index(
        "ix_tenants_vector_correlation_id",
        "tenants",
        ["vector_correlation_id"],
        schema=schema,
    )
    op.execute(
        sa.text(
            f'INSERT INTO "{schema}"."tenants" (tenant_id, tier) '
            + "VALUES (:tid, 'dedicated') ON CONFLICT DO NOTHING"
        ).bindparams(tid=DEFAULT_TENANT_ID)
    )

    # 2. Add tenant_id everywhere + backfill + NOT NULL on the tenant-scoped set.
    for tname in TENANT_SCOPED:
        if not column_exists(tname, "tenant_id"):
            op.add_column(
                tname, sa.Column("tenant_id", sa.TEXT(), nullable=True), schema=schema
            )
        op.execute(
            sa.text(
                f'UPDATE "{schema}"."{tname}" SET tenant_id = :tid '
                + "WHERE tenant_id IS NULL"
            ).bindparams(tid=DEFAULT_TENANT_ID)
        )
        op.alter_column(tname, "tenant_id", nullable=False, schema=schema)

    # 3. Drop ALL old FKs first (they block the PK/unique changes below), incl.
    #    queue.session_id -> sessions.id from an earlier revision.
    _drop_all_fks((*TENANT_SCOPED, "queue"))

    # 4. Rebuild primary keys -> tenant-leading composite.
    for tname, cols in NEW_PKS.items():
        _drop_pk(tname)
        op.create_primary_key(f"pk_{tname}", tname, cols, schema=schema)

    # 5. Drop old uniques, add the new tenant-scoped ones.
    for tname in TENANT_SCOPED:
        _drop_all_uniques(tname)
    for tname, uniques in NEW_UNIQUES.items():
        for uname, cols in uniques:
            op.create_unique_constraint(uname, tname, cols, schema=schema)

    # 6. tenant_id -> tenants.tenant_id on every tenant-scoped table, then the
    #    composite natural-key FKs (their targets — the new uniques — now exist).
    for tname in TENANT_SCOPED:
        op.create_foreign_key(
            f"fk_{tname}_tenant_id_tenants",
            tname,
            "tenants",
            ["tenant_id"],
            ["tenant_id"],
            source_schema=schema,
            referent_schema=schema,
        )
    for name, src, local_cols, ref, ref_cols, ondelete in NEW_FKS:
        op.create_foreign_key(
            name,
            src,
            ref,
            local_cols,
            ref_cols,
            ondelete=ondelete,
            source_schema=schema,
            referent_schema=schema,
        )

    # 7. Service tables: tenant_id is plain attribution — nullable, no FK, no RLS.
    for tname in ("queue", "active_queue_sessions"):
        if not column_exists(tname, "tenant_id"):
            op.add_column(
                tname, sa.Column("tenant_id", sa.TEXT(), nullable=True), schema=schema
            )
    if not index_exists("queue", "ix_queue_tenant_id"):
        op.create_index("ix_queue_tenant_id", "queue", ["tenant_id"], schema=schema)

    # 8. New tenant-scoped / vector / fts indexes (idempotent).
    _create_new_indexes()


def _create_new_indexes() -> None:
    def _mk(name: str, table: str, cols: list[Any], **kw: Any) -> None:
        if not index_exists(table, name):
            op.create_index(name, table, cols, schema=schema, **kw)

    _mk("ix_peers_tenant_workspace", "peers", ["tenant_id", "workspace_name"])
    _mk("ix_sessions_tenant_workspace", "sessions", ["tenant_id", "workspace_name"])
    _mk(
        "ix_messages_session_lookup",
        "messages",
        ["tenant_id", "session_name", "id"],
        postgresql_include=["created_at"],
    )
    _mk(
        "ix_messages_peer_lookup",
        "messages",
        ["tenant_id", "workspace_name", "peer_name", "created_at"],
    )
    _mk(
        "ix_messages_content_gin",
        "messages",
        [sa.text("to_tsvector('english', content)")],
        postgresql_using="gin",
    )
    _mk(
        "ix_message_embeddings_message_tenant",
        "message_embeddings",
        ["message_id", "tenant_id"],
    )
    _mk(
        "ix_message_embeddings_embedding_hnsw",
        "message_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    _mk(
        "ix_message_embeddings_sync_state_last_sync_at",
        "message_embeddings",
        ["sync_state", "last_sync_at"],
    )
    _mk(
        "ix_documents_tenant_collection",
        "documents",
        ["tenant_id", "observer", "observed", "workspace_name"],
    )
    _mk(
        "ix_documents_embedding_hnsw",
        "documents",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    _mk("ix_documents_source_ids_gin", "documents", ["source_ids"], postgresql_using="gin")
    _mk(
        "ix_documents_sync_state_last_sync_at",
        "documents",
        ["sync_state", "last_sync_at"],
    )
    _mk(
        "ix_webhook_endpoints_tenant_workspace",
        "webhook_endpoints",
        ["tenant_id", "workspace_name"],
    )


def downgrade() -> None:
    # Reverse the transform: drop new FKs/uniques/indexes, restore sole-id PKs,
    # drop tenant_id, drop tenants. Dropping the tenant_id column cascades away any
    # residual constraint that references it.
    if not table_exists("tenants"):
        return

    for name, src, *_ in NEW_FKS:
        insp = _inspector()
        if any(fk.get("name") == name for fk in insp.get_foreign_keys(src, schema=schema)):
            op.drop_constraint(name, src, type_="foreignkey", schema=schema)
    for tname in TENANT_SCOPED:
        insp = _inspector()
        fk_name = f"fk_{tname}_tenant_id_tenants"
        if any(fk.get("name") == fk_name for fk in insp.get_foreign_keys(tname, schema=schema)):
            op.drop_constraint(fk_name, tname, type_="foreignkey", schema=schema)

    for tname, uniques in NEW_UNIQUES.items():
        for uname, _cols in uniques:
            insp = _inspector()
            if any(
                uq.get("name") == uname
                for uq in insp.get_unique_constraints(tname, schema=schema)
            ):
                op.drop_constraint(uname, tname, type_="unique", schema=schema)

    # Restore sole-id / natural PKs, then drop tenant_id (with any index on it).
    old_pks = {
        "session_peers": ["workspace_name", "session_name", "peer_name"],
    }
    for tname in TENANT_SCOPED:
        _drop_pk(tname)
        cols = old_pks.get(tname, ["id"])
        op.create_primary_key(f"pk_{tname}", tname, cols, schema=schema)

    for tname in (*TENANT_SCOPED, "queue", "active_queue_sessions"):
        if column_exists(tname, "tenant_id"):
            op.drop_column(tname, "tenant_id", schema=schema)

    op.drop_table("tenants", schema=schema)
