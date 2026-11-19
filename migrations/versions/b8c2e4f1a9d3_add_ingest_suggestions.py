"""add ingest suggestions

`ingest_suggestions` holds what the organizer suggests about an import for a
person to confirm: for now, two accounts that look like one person, with the
evidence, the stage that made it and whether a person accepted or dismissed
it. Tenant-scoped like the other ingest tables (tenant_id leads the primary key
and the unique key), minus physical partitioning, with no foreign key to
peers or workspaces: workspace deletion removes its rows explicitly. DDL only:
there is nothing to backfill.

Revision ID: b8c2e4f1a9d3
Revises: 7dbf160eb7c9
Create Date: 2026-11-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "b8c2e4f1a9d3"
down_revision: str | None = "7dbf160eb7c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add ingest_suggestions; skipped when it already exists, so it can be rerun."""
    inspector = sa.inspect(op.get_bind())
    if table_exists("ingest_suggestions", inspector):
        return
    op.create_table(
        "ingest_suggestions",
        sa.Column("tenant_id", sa.TEXT, nullable=False),
        sa.Column("id", sa.TEXT, nullable=False),
        sa.Column("workspace_name", sa.TEXT, nullable=False),
        sa.Column("kind", sa.TEXT, nullable=False),
        sa.Column("subject_key", sa.TEXT, nullable=False),
        sa.Column("accounts", JSONB, nullable=False),
        sa.Column("confidence", sa.TEXT, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("evidence", JSONB, nullable=False),
        sa.Column("stage", sa.TEXT, nullable=False),
        sa.Column("rules_version", sa.Integer, nullable=False),
        sa.Column(
            "organizer_tokens", sa.Integer, nullable=False, server_default=sa.text("0")
        ),
        sa.Column("status", sa.TEXT, nullable=False),
        sa.Column(
            "refreshed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
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
            "kind",
            "subject_key",
            name="uq_ingest_suggestions_subject",
        ),
        sa.CheckConstraint("kind IN ('same_person')", name="kind_valid"),
        sa.CheckConstraint("confidence IN ('high', 'low')", name="confidence_valid"),
        sa.CheckConstraint(
            "status IN ('open', 'accepted', 'dismissed')", name="status_valid"
        ),
        schema=schema,
    )


def downgrade() -> None:
    """Drop ingest_suggestions, if present."""
    if table_exists("ingest_suggestions", sa.inspect(op.get_bind())):
        op.drop_table("ingest_suggestions", schema=schema)
