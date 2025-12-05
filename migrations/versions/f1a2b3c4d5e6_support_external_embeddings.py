"""add chunk_index to message_embeddings, make embeddings nullable, add soft delete

This migration:
1. Adds the chunk_index column to message_embeddings table for tracking
   chunked message embeddings in external vector stores (turbopuffer/lancedb).
2. Makes embedding columns nullable in both message_embeddings and documents tables
   since embeddings are now stored in external vector stores instead of PostgreSQL.
3. Adds deleted_at column to documents table for soft delete support, enabling
   hybrid sync/soft delete pattern for vector store consistency.

Revision ID: f1a2b3c4d5e6
Revises: baa22cad81e2
Create Date: 2025-11-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import column_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "baa22cad81e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add chunk_index, make embeddings nullable, add deleted_at for soft delete."""
    inspector = sa.inspect(op.get_bind())

    # Add chunk_index column to message_embeddings if it doesn't exist
    # This is needed to track which chunk of a message this embedding represents
    # Vector ID format: {message_public_id}_{chunk_index}
    if not column_exists("message_embeddings", "chunk_index", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "chunk_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )

    # Make message_embeddings.embedding nullable since embeddings are now stored
    # in external vector stores (turbopuffer/lancedb) instead of PostgreSQL
    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(1536),
        nullable=True,
        schema=schema,
    )

    # Make documents.embedding nullable for the same reason
    # (this should already be nullable, but ensure it for consistency)
    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(1536),
        nullable=True,
        schema=schema,
    )

    # Add deleted_at column to documents for soft delete support
    # This enables hybrid sync/soft delete pattern:
    # - Try to delete from vector store first
    # - If successful, hard delete from DB
    # - If vector delete fails, soft delete (set deleted_at) and let cleanup job handle it
    if not column_exists("documents", "deleted_at", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "deleted_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            schema=schema,
        )
        # Create partial index for efficient cleanup queries (only index non-null values)
        op.create_index(
            "ix_documents_deleted_at",
            "documents",
            ["deleted_at"],
            schema=schema,
            postgresql_where=sa.text("deleted_at IS NOT NULL"),
        )


def downgrade() -> None:
    """Remove chunk_index, deleted_at columns and revert embedding columns."""
    inspector = sa.inspect(op.get_bind())

    # Remove deleted_at column and index from documents
    if column_exists("documents", "deleted_at", inspector):
        op.drop_index("ix_documents_deleted_at", table_name="documents", schema=schema)
        op.drop_column("documents", "deleted_at", schema=schema)

    # Revert documents.embedding back to nullable=True (it was originally nullable=True)
    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(1536),
        nullable=True,  # Keep as nullable since it was nullable in the original schema
        schema=schema,
    )

    # Revert message_embeddings.embedding back to nullable=False
    # Note: This may fail if there are NULL values in the database
    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(1536),
        nullable=False,
        schema=schema,
    )

    # Remove chunk_index column if it exists
    if column_exists("message_embeddings", "chunk_index", inspector):
        op.drop_column("message_embeddings", "chunk_index", schema=schema)
