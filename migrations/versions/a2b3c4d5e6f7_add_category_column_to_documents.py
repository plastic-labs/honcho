"""add_category_column_to_documents

Add nullable category column to documents table for scoped observation
retrieval. Enables filtering observations by semantic category (e.g.,
"preferences", "business_context", "personal") for more precise search.

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-04-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "a2b3c4d5e6f7"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add category column with index for scoped observation retrieval."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not column_exists("documents", "category", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "category",
                sa.TEXT(),
                nullable=True,
            ),
            schema=schema,
        )

    if not index_exists("documents", "ix_documents_category", inspector):
        op.create_index(
            "ix_documents_category",
            "documents",
            ["category"],
            schema=schema,
        )


def downgrade() -> None:
    """Remove category column and index."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if index_exists("documents", "ix_documents_category", inspector):
        op.drop_index("ix_documents_category", table_name="documents", schema=schema)
    if column_exists("documents", "category", inspector):
        op.drop_column("documents", "category", schema=schema)
