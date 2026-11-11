"""add ingest tables

`ingest_records` holds one row per source record received through the ingest API,
unique on its source identity, with the durable `derivation` state the ingest
enqueuer drains. `ingest_session_bindings` and `ingest_peer_bindings` map external
containers and people to the sessions and peers ingest created for them, with
the newest display name seen for each and, once a person confirms a link or move,
when they did. Each record keeps the peer bindings of its
author and participants, so linked people can be split again.
`workspaces.auto_enabled_at` records when a workspace opted into auto organization.

The three tables are tenant-scoped like the other data tables (tenant_id leads the
primary key and every unique key), minus physical partitioning. They have no
foreign key to sessions, peers or workspaces: deletion removes their rows
explicitly, under the ingest fence. DDL only: there is nothing to backfill.

Revision ID: 7dbf160eb7c9
Revises: 4f9c2a7e8d31
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import column_exists, get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "7dbf160eb7c9"
down_revision: str | None = "4f9c2a7e8d31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not column_exists("workspaces", "auto_enabled_at", inspector):
        op.add_column(
            "workspaces",
            sa.Column("auto_enabled_at", sa.DateTime(timezone=True), nullable=True),
            schema=schema,
        )

    if not table_exists("ingest_records", inspector):
        op.create_table(
            "ingest_records",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("source", sa.TEXT, nullable=False),
            sa.Column("source_instance", sa.TEXT, nullable=False),
            sa.Column("identity_kind", sa.TEXT, nullable=False),
            sa.Column("identity", sa.TEXT, nullable=False),
            sa.Column("message_public_id", sa.TEXT, nullable=True),
            sa.Column("session_name", sa.TEXT, nullable=False),
            sa.Column("content_sha256", sa.TEXT, nullable=False),
            sa.Column("revised_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("configuration", JSONB, nullable=True),
            sa.Column(
                "reasoning_disabled",
                sa.Boolean,
                nullable=False,
                server_default=sa.text("false"),
            ),
            sa.Column("derivation", sa.TEXT, nullable=False),
            sa.Column("author_binding_id", sa.TEXT, nullable=True),
            sa.Column("participant_binding_ids", JSONB, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.UniqueConstraint(
                "tenant_id",
                "workspace_name",
                "source",
                "source_instance",
                "identity_kind",
                "identity",
                name="uq_ingest_records_identity",
            ),
            sa.CheckConstraint(
                "identity_kind IN ('source_id', 'content_hash')",
                name="identity_kind_valid",
            ),
            sa.CheckConstraint(
                "derivation IN ('pending', 'queued', 'deferred')",
                name="derivation_valid",
            ),
            schema=schema,
        )
        op.create_index(
            "ix_ingest_records_session",
            "ingest_records",
            ["tenant_id", "workspace_name", "session_name"],
            schema=schema,
        )
        op.create_index(
            "ix_ingest_records_pending",
            "ingest_records",
            ["tenant_id", "workspace_name", "session_name"],
            postgresql_where=sa.text("derivation = 'pending'"),
            schema=schema,
        )
        op.create_index(
            "ix_ingest_records_author_binding",
            "ingest_records",
            ["tenant_id", "workspace_name", "author_binding_id"],
            postgresql_where=sa.text("author_binding_id IS NOT NULL"),
            schema=schema,
        )

    if not table_exists("ingest_session_bindings", inspector):
        op.create_table(
            "ingest_session_bindings",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("source", sa.TEXT, nullable=False),
            sa.Column("source_instance", sa.TEXT, nullable=False),
            sa.Column("key_kind", sa.TEXT, nullable=False),
            sa.Column("key_value", sa.TEXT, nullable=False),
            sa.Column("session_name", sa.TEXT, nullable=False),
            sa.Column("display_name", sa.TEXT, nullable=True),
            sa.Column("display_name_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.UniqueConstraint(
                "tenant_id",
                "workspace_name",
                "source",
                "source_instance",
                "key_kind",
                "key_value",
                name="uq_ingest_session_bindings_key",
            ),
            sa.CheckConstraint(
                "key_kind IN ('container', 'standalone')", name="key_kind_valid"
            ),
            schema=schema,
        )
        op.create_index(
            "ix_ingest_session_bindings_session",
            "ingest_session_bindings",
            ["tenant_id", "workspace_name", "session_name"],
            schema=schema,
        )

    if not table_exists("ingest_peer_bindings", inspector):
        op.create_table(
            "ingest_peer_bindings",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("source", sa.TEXT, nullable=False),
            sa.Column("source_instance", sa.TEXT, nullable=False),
            sa.Column("key_kind", sa.TEXT, nullable=False),
            sa.Column("key_value", sa.TEXT, nullable=False),
            sa.Column("peer_name", sa.TEXT, nullable=False),
            sa.Column("display_name", sa.TEXT, nullable=True),
            sa.Column("display_name_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.UniqueConstraint(
                "tenant_id",
                "workspace_name",
                "source",
                "source_instance",
                "key_kind",
                "key_value",
                name="uq_ingest_peer_bindings_key",
            ),
            sa.CheckConstraint(
                "key_kind IN ('external_id', 'provisional_name', 'source_peer')",
                name="key_kind_valid",
            ),
            schema=schema,
        )
        op.create_index(
            "ix_ingest_peer_bindings_peer",
            "ingest_peer_bindings",
            ["tenant_id", "workspace_name", "peer_name"],
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    for table in ("ingest_peer_bindings", "ingest_session_bindings", "ingest_records"):
        if table_exists(table, inspector):
            op.drop_table(table, schema=schema)
    if column_exists("workspaces", "auto_enabled_at", inspector):
        op.drop_column("workspaces", "auto_enabled_at", schema=schema)
