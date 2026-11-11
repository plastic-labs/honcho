"""add dream history tables

`dream_runs` holds one row per dream and `dream_run_changes` one row per change
a dream committed: conclusions created, conclusions removed, and peer-card
updates. Both are tenant-scoped like the other data tables (tenant_id leads the
primary key and every foreign key), minus physical partitioning. DDL only:
dreams before this revision recorded nothing to backfill.

Revision ID: a7e3c1d9f2b4
Revises: d4b8e2f7a913
Create Date: 2026-11-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "a7e3c1d9f2b4"
down_revision: str | None = "d4b8e2f7a913"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not table_exists("dream_runs", inspector):
        op.create_table(
            "dream_runs",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("observer", sa.TEXT, nullable=False),
            sa.Column("observed", sa.TEXT, nullable=False),
            sa.Column("dream_type", sa.TEXT, nullable=True),
            sa.Column("trigger_reason", sa.TEXT, nullable=True),
            sa.Column(
                "status", sa.TEXT, nullable=False, server_default=sa.text("'running'")
            ),
            sa.Column("created_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column("removed_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column(
                "peer_card_updated",
                sa.Boolean,
                nullable=False,
                server_default=sa.text("false"),
            ),
            sa.Column(
                "started_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("pruned_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.ForeignKeyConstraint(
                ["workspace_name", "tenant_id"],
                [f"{schema}.workspaces.name", f"{schema}.workspaces.tenant_id"],
                name="fk_dream_runs_ws_tenant_workspaces",
            ),
            sa.CheckConstraint(
                "status IN ('running', 'completed', 'failed', 'interrupted')",
                name="status_valid",
            ),
            schema=schema,
        )
        op.create_index(
            "ix_dream_runs_tenant_workspace_observed_started",
            "dream_runs",
            ["tenant_id", "workspace_name", "observed", "started_at", "id"],
            schema=schema,
        )
        op.create_index(
            "ix_dream_runs_started_at", "dream_runs", ["started_at"], schema=schema
        )

    if not table_exists("dream_run_changes", inspector):
        op.create_table(
            "dream_run_changes",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.BigInteger, sa.Identity(), nullable=False),
            sa.Column("run_id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("kind", sa.TEXT, nullable=False),
            sa.Column("specialist", sa.TEXT, nullable=True),
            sa.Column("conclusion_id", sa.TEXT, nullable=True),
            sa.Column("level", sa.TEXT, nullable=True),
            sa.Column("content", sa.TEXT, nullable=True),
            sa.Column("session_name", sa.TEXT, nullable=True),
            sa.Column("source_ids", JSONB, nullable=True),
            sa.Column("replaced_by", sa.TEXT, nullable=True),
            sa.Column("peer_card_before", JSONB, nullable=True),
            sa.Column("peer_card_after", JSONB, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("tenant_id", "id"),
            sa.ForeignKeyConstraint(["tenant_id"], [f"{schema}.tenants.tenant_id"]),
            sa.ForeignKeyConstraint(
                ["run_id", "tenant_id"],
                [f"{schema}.dream_runs.id", f"{schema}.dream_runs.tenant_id"],
                ondelete="CASCADE",
                name="fk_dream_run_changes_run_tenant_dream_runs",
            ),
            sa.ForeignKeyConstraint(
                ["workspace_name", "tenant_id"],
                [f"{schema}.workspaces.name", f"{schema}.workspaces.tenant_id"],
                name="fk_dream_run_changes_ws_tenant_workspaces",
            ),
            sa.CheckConstraint(
                "kind IN ('created', 'removed', 'peer_card')", name="kind_valid"
            ),
            schema=schema,
        )
        op.create_index(
            "ix_dream_run_changes_tenant_run_id",
            "dream_run_changes",
            ["tenant_id", "run_id", "id"],
            schema=schema,
        )
        op.create_index(
            "ix_dream_run_changes_removed_conclusion",
            "dream_run_changes",
            ["tenant_id", "workspace_name", "conclusion_id"],
            postgresql_where=sa.text("kind = 'removed'"),
            schema=schema,
        )
        op.create_index(
            "ix_dream_run_changes_session",
            "dream_run_changes",
            ["tenant_id", "workspace_name", "session_name"],
            postgresql_where=sa.text("session_name IS NOT NULL"),
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_exists("dream_run_changes", inspector):
        op.drop_table("dream_run_changes", schema=schema)
    if table_exists("dream_runs", inspector):
        op.drop_table("dream_runs", schema=schema)
