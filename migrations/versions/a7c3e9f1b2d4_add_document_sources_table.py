"""add document_sources table

Normalize reasoning-tree linkage into a document_sources edge table. This
revision is DDL only. Existing linkage in documents.source_ids and the legacy
internal_metadata->'source_ids' / 'premise_ids' locations is drained into the
new table by the deriver's reconciler (backfill_document_sources), so the
api pod's init container is not blocked on a full-table copy.

The source_ids column and its GIN index stay until the drain completes; a
follow-up migration drops both.

Revision ID: a7c3e9f1b2d4
Revises: e4eba9cfaa6f
Create Date: 2026-08-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "a7c3e9f1b2d4"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not table_exists("document_sources", inspector):
        op.create_table(
            "document_sources",
            sa.Column("derived_id", sa.TEXT, nullable=False),
            sa.Column("source_id", sa.TEXT, nullable=False),
            sa.Column("position", sa.Integer, nullable=False, server_default="0"),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.PrimaryKeyConstraint("derived_id", "source_id"),
            sa.ForeignKeyConstraint(
                ["derived_id"],
                [f"{schema}.documents.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["workspace_name"],
                [f"{schema}.workspaces.name"],
            ),
            sa.CheckConstraint("length(source_id) = 21", name="source_id_length"),
            sa.CheckConstraint(
                "source_id ~ '^[A-Za-z0-9_-]+$'", name="source_id_format"
            ),
            schema=schema,
        )
        op.create_index(
            "ix_document_sources_source_id",
            "document_sources",
            ["source_id", "workspace_name"],
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Reassemble JSONB arrays for rows the drain moved or new code wrote;
    # undrained rows still hold their original column value.
    op.execute(f"""
        UPDATE {schema}.documents d
        SET source_ids = links.ids
        FROM (
            SELECT derived_id, jsonb_agg(source_id ORDER BY position) AS ids
            FROM {schema}.document_sources
            GROUP BY derived_id
        ) links
        WHERE d.id = links.derived_id
    """)

    if table_exists("document_sources", inspector):
        op.drop_table("document_sources", schema=schema)
