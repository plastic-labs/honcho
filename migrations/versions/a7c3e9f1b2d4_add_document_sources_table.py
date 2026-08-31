"""add document_sources table, backfill from JSONB source_ids

Normalize reasoning-tree linkage into a document_sources edge table.
Backfills from both the documents.source_ids column and the legacy
internal_metadata->'source_ids' location, dropping entries that are not
well-formed 21-char nanoids (they never resolved to documents anyway).

The old source_ids column is kept (unwritten) for one release as a
rollback net; a follow-up migration drops it.

Revision ID: a7c3e9f1b2d4
Revises: e4eba9cfaa6f
Create Date: 2026-08-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists, table_exists

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

    # Backfill: column takes precedence over legacy internal_metadata
    # (mirrors the old resolved_source_ids property). Malformed entries
    # are dropped; DISTINCT ON dedupes repeated IDs within one document.
    op.execute(f"""
        INSERT INTO {schema}.document_sources
            (derived_id, source_id, position, workspace_name)
        SELECT DISTINCT ON (d.id, s.value)
            d.id, s.value, s.ord - 1, d.workspace_name
        FROM {schema}.documents d
        CROSS JOIN LATERAL jsonb_array_elements_text(
            CASE
                WHEN jsonb_typeof(d.source_ids) = 'array'
                    THEN d.source_ids
                WHEN jsonb_typeof(d.internal_metadata->'source_ids') = 'array'
                    THEN d.internal_metadata->'source_ids'
                WHEN jsonb_typeof(d.internal_metadata->'premise_ids') = 'array'
                    THEN d.internal_metadata->'premise_ids'
                ELSE '[]'::jsonb
            END
        ) WITH ORDINALITY AS s(value, ord)
        WHERE s.value ~ '^[A-Za-z0-9_-]{{21}}$'
        ORDER BY d.id, s.value, s.ord
        ON CONFLICT DO NOTHING
    """)

    # The ORM no longer queries source_ids; drop its GIN index. The column
    # itself stays for one release as a rollback net.
    if index_exists("documents", "ix_documents_source_ids_gin", inspector):
        op.drop_index(
            "ix_documents_source_ids_gin",
            table_name="documents",
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Reassemble JSONB arrays for rows written after the upgrade (the
    # retained column still holds pre-upgrade data for older rows).
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

    if not index_exists("documents", "ix_documents_source_ids_gin", inspector):
        op.create_index(
            "ix_documents_source_ids_gin",
            "documents",
            ["source_ids"],
            postgresql_using="gin",
            schema=schema,
        )

    if table_exists("document_sources", inspector):
        op.drop_table("document_sources", schema=schema)
