"""ingest: the end of the dry run, tombstones, and sync states

Three changes for following a source live:

- `workspaces.ingest_derived_at`: when an auto workspace started deriving, which
  ends its dry run (edits and deletes are refused after it, and `derive: "auto"`
  queues). Backfilled for auto workspaces that already hold derived or queued
  work: a non-deferred ingest record, a queue item or a collection.
- `ingest_records` keeps a tombstone for a deleted record: `derivation` may be
  `deleted`, and a tombstone for a record never seen has no session or content
  hash. A CHECK keeps both on every other record. The CHECKs are added NOT VALID
  and validated after, so the table isn't locked against writes while it's read.
- `ingest_sync_states`: where a connector following a source has got to, and its
  health. Tenant-scoped like the other ingest tables (tenant_id leads the primary
  key and the unique key), minus physical partitioning, with no foreign key but
  tenants: workspace deletion removes its rows explicitly.

Revision ID: c4d9e1a7b2f6
Revises: b8c2e4f1a9d3
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import (
    column_exists,
    constraint_exists,
    get_schema,
    table_exists,
)

# revision identifiers, used by Alembic.
revision: str = "c4d9e1a7b2f6"
down_revision: str | None = "b8c2e4f1a9d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

# Full names, as the models' naming convention renders them (src/db.py).
DERIVATION_VALID = "ck_ingest_records_derivation_valid"
TOMBSTONE_OR_COMPLETE = "ck_ingest_records_tombstone_or_complete"


def upgrade() -> None:
    """Apply all three; each step is skipped when already done, so it can be rerun."""
    inspector = sa.inspect(op.get_bind())

    if not column_exists("workspaces", "ingest_derived_at", inspector):
        op.add_column(
            "workspaces",
            sa.Column("ingest_derived_at", sa.DateTime(timezone=True), nullable=True),
            schema=schema,
        )
    op.execute(
        sa.text(
            f"""
            UPDATE {schema}.workspaces w SET ingest_derived_at = now()
            WHERE w.auto_enabled_at IS NOT NULL AND w.ingest_derived_at IS NULL
              AND (
                EXISTS (SELECT 1 FROM {schema}.ingest_records r
                        WHERE r.tenant_id = w.tenant_id
                          AND r.workspace_name = w.name
                          AND r.derivation IN ('pending', 'queued'))
                OR EXISTS (SELECT 1 FROM {schema}.queue q
                           WHERE q.tenant_id IS NOT DISTINCT FROM w.tenant_id
                             AND q.workspace_name = w.name)
                OR EXISTS (SELECT 1 FROM {schema}.collections c
                           WHERE c.tenant_id = w.tenant_id
                             AND c.workspace_name = w.name)
              )
            """
        )
    )

    op.alter_column("ingest_records", "session_name", nullable=True, schema=schema)
    op.alter_column("ingest_records", "content_sha256", nullable=True, schema=schema)
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records DROP CONSTRAINT IF EXISTS"
            + f" {DERIVATION_VALID}"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records ADD CONSTRAINT {DERIVATION_VALID}"
            + " CHECK (derivation IN ('pending', 'queued', 'deferred', 'deleted'))"
            + " NOT VALID"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records VALIDATE CONSTRAINT {DERIVATION_VALID}"
        )
    )
    if not constraint_exists("ingest_records", TOMBSTONE_OR_COMPLETE, "check"):
        op.execute(
            sa.text(
                f"ALTER TABLE {schema}.ingest_records ADD CONSTRAINT"
                + f" {TOMBSTONE_OR_COMPLETE} CHECK (derivation = 'deleted'"
                + " OR (session_name IS NOT NULL AND content_sha256 IS NOT NULL))"
                + " NOT VALID"
            )
        )
        op.execute(
            sa.text(
                f"ALTER TABLE {schema}.ingest_records VALIDATE CONSTRAINT"
                + f" {TOMBSTONE_OR_COMPLETE}"
            )
        )

    if not table_exists("ingest_sync_states", inspector):
        op.create_table(
            "ingest_sync_states",
            sa.Column("tenant_id", sa.TEXT, nullable=False),
            sa.Column("id", sa.TEXT, nullable=False),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.Column("source", sa.TEXT, nullable=False),
            sa.Column("source_instance", sa.TEXT, nullable=False),
            sa.Column("position", JSONB, nullable=False),
            sa.Column("revision", sa.TEXT, nullable=False),
            sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_record_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error", sa.TEXT, nullable=True),
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
                name="uq_ingest_sync_states_source",
            ),
            schema=schema,
        )


def downgrade() -> None:
    """Drop sync states and the marker; tombstones are deleted to restore the checks."""
    inspector = sa.inspect(op.get_bind())
    if table_exists("ingest_sync_states", inspector):
        op.drop_table("ingest_sync_states", schema=schema)
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records DROP CONSTRAINT IF EXISTS"
            + f" {TOMBSTONE_OR_COMPLETE}"
        )
    )
    op.execute(
        sa.text(f"DELETE FROM {schema}.ingest_records WHERE derivation = 'deleted'")
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records DROP CONSTRAINT IF EXISTS"
            + f" {DERIVATION_VALID}"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {schema}.ingest_records ADD CONSTRAINT {DERIVATION_VALID}"
            + " CHECK (derivation IN ('pending', 'queued', 'deferred'))"
        )
    )
    op.alter_column("ingest_records", "content_sha256", nullable=False, schema=schema)
    op.alter_column("ingest_records", "session_name", nullable=False, schema=schema)
    if column_exists("workspaces", "ingest_derived_at", inspector):
        op.drop_column("workspaces", "ingest_derived_at", schema=schema)
