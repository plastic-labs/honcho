"""add document superseded_by column

Add superseded_by column to documents table for tracking document
supersession relationships. When a document is replaced by a more
informative version (via deduplication or specialist knowledge updates),
the old document's superseded_by field points to its replacement.

Revision ID: a3f7b9d2c1e4
Revises: e4eba9cfaa6f
Create Date: 2026-03-27

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "a3f7b9d2c1e4"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add superseded_by column and partial index to documents table."""
    inspector = sa.inspect(op.get_bind())

    if not column_exists("documents", "superseded_by", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "superseded_by",
                sa.TEXT(),
                nullable=True,
            ),
            schema=schema,
        )

    # Partial index: only index rows where superseded_by is set.
    # Used by reconciler to efficiently find superseded tombstones.
    if not index_exists("documents", "ix_documents_superseded_by", inspector):
        op.create_index(
            "ix_documents_superseded_by",
            "documents",
            ["superseded_by"],
            schema=schema,
            postgresql_where=sa.text("superseded_by IS NOT NULL"),
        )


def downgrade() -> None:
    """Remove superseded_by column and index from documents table."""
    inspector = sa.inspect(op.get_bind())

    if index_exists("documents", "ix_documents_superseded_by", inspector):
        op.drop_index(
            "ix_documents_superseded_by",
            table_name="documents",
            schema=schema,
        )

    if column_exists("documents", "superseded_by", inspector):
        op.drop_column("documents", "superseded_by", schema=schema)
