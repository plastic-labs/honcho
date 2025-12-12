"""add_reasoning_tree_columns

Add premise_ids and source_ids columns to documents table for
reasoning tree traversal. These enable linking deductive conclusions
to their premise observations and inductive patterns to their source
observations.

Revision ID: f1a2b3c4d5e6
Revises: 66e08642ecd6
Create Date: 2025-12-11

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "66e08642ecd6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add premise_ids and source_ids columns with GIN indexes for tree traversal."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Add premise_ids column (nullable, for deductive observations)
    if not column_exists("documents", "premise_ids", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "premise_ids",
                JSONB,
                nullable=True,
                server_default=sa.text("NULL"),
            ),
            schema=schema,
        )

    # Add source_ids column (nullable, for inductive observations)
    if not column_exists("documents", "source_ids", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "source_ids",
                JSONB,
                nullable=True,
                server_default=sa.text("NULL"),
            ),
            schema=schema,
        )

    # Add GIN index on premise_ids for efficient child lookups
    # (finding all observations that have a given observation as a premise)
    if not index_exists("documents", "ix_documents_premise_ids_gin", inspector):
        op.create_index(
            "ix_documents_premise_ids_gin",
            "documents",
            ["premise_ids"],
            postgresql_using="gin",
            schema=schema,
        )

    # Add GIN index on source_ids for efficient child lookups
    # (finding all observations that have a given observation as a source)
    if not index_exists("documents", "ix_documents_source_ids_gin", inspector):
        op.create_index(
            "ix_documents_source_ids_gin",
            "documents",
            ["source_ids"],
            postgresql_using="gin",
            schema=schema,
        )


def downgrade() -> None:
    """Remove premise_ids and source_ids columns and their indexes."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Drop GIN indexes first
    if index_exists("documents", "ix_documents_premise_ids_gin", inspector):
        op.drop_index(
            "ix_documents_premise_ids_gin",
            table_name="documents",
            schema=schema,
        )

    if index_exists("documents", "ix_documents_source_ids_gin", inspector):
        op.drop_index(
            "ix_documents_source_ids_gin",
            table_name="documents",
            schema=schema,
        )

    # Drop columns
    if column_exists("documents", "premise_ids", inspector):
        op.drop_column("documents", "premise_ids", schema=schema)

    if column_exists("documents", "source_ids", inspector):
        op.drop_column("documents", "source_ids", schema=schema)
