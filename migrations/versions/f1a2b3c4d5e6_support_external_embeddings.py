"""make embeddings nullable, add soft delete, add vector sync state

This migration:
1. Makes embedding columns nullable in both message_embeddings and documents tables
   since embeddings are now stored in external vector stores instead of PostgreSQL.
2. Adds deleted_at column to documents table for soft delete support, enabling
   hybrid sync/soft delete pattern for vector store consistency.
3. Adds sync_state, last_sync_at, and sync_attempts columns to documents and
   message_embeddings tables for tracking vector store synchronization status.

Revision ID: f1a2b3c4d5e6
Revises: 110bdf470272
Create Date: 2025-11-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "110bdf470272"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Make embeddings nullable, add deleted_at, add sync state."""
    inspector = sa.inspect(op.get_bind())

    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(1536),
        nullable=True,
        schema=schema,
    )

    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(1536),
        nullable=True,
        schema=schema,
    )

    # Add deleted_at column to documents for soft delete support
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

    # Add sync state columns to documents table
    if not column_exists("documents", "sync_state", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "sync_state",
                sa.TEXT(),
                nullable=False,
                server_default="pending",  # Existing records need reconciliation
            ),
            schema=schema,
        )
        op.create_index(
            "ix_documents_sync_state",
            "documents",
            ["sync_state"],
            schema=schema,
        )

    if not column_exists("documents", "last_sync_at", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "last_sync_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            schema=schema,
        )

    if not column_exists("documents", "sync_attempts", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "sync_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )

    # Add composite index for efficient reconciliation queries after both columns exist
    # Reconciliation orders by: WHERE sync_state='pending' ORDER BY last_sync_at
    if not index_exists("documents", "ix_documents_sync_state_last_sync_at", inspector):
        op.create_index(
            "ix_documents_sync_state_last_sync_at",
            "documents",
            ["sync_state", "last_sync_at"],
            schema=schema,
        )

    # Add sync state columns to message_embeddings table
    if not column_exists("message_embeddings", "sync_state", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "sync_state",
                sa.TEXT(),
                nullable=False,
                server_default="pending",  # Existing records need reconciliation
            ),
            schema=schema,
        )
        op.create_index(
            "ix_message_embeddings_sync_state",
            "message_embeddings",
            ["sync_state"],
            schema=schema,
        )

    if not column_exists("message_embeddings", "last_sync_at", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "last_sync_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            schema=schema,
        )

    if not column_exists("message_embeddings", "sync_attempts", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "sync_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )

    # Add composite index for efficient reconciliation queries after both columns exist
    # Reconciliation orders by: WHERE sync_state='pending' ORDER BY last_sync_at
    if not index_exists(
        "message_embeddings", "ix_message_embeddings_sync_state_last_sync_at", inspector
    ):
        op.create_index(
            "ix_message_embeddings_sync_state_last_sync_at",
            "message_embeddings",
            ["sync_state", "last_sync_at"],
            schema=schema,
        )


def downgrade() -> None:
    """Remove deleted_at columns and revert embedding columns."""
    inspector = sa.inspect(op.get_bind())

    # Remove sync state columns from message_embeddings
    if column_exists("message_embeddings", "sync_attempts", inspector):
        op.drop_column("message_embeddings", "sync_attempts", schema=schema)

    if column_exists("message_embeddings", "last_sync_at", inspector):
        op.drop_column("message_embeddings", "last_sync_at", schema=schema)

    if column_exists("message_embeddings", "sync_state", inspector):
        # Drop composite index first
        op.drop_index(
            "ix_message_embeddings_sync_state_last_sync_at",
            table_name="message_embeddings",
            schema=schema,
        )
        op.drop_index(
            "ix_message_embeddings_sync_state",
            table_name="message_embeddings",
            schema=schema,
        )
        op.drop_column("message_embeddings", "sync_state", schema=schema)

    # Remove sync state columns from documents
    if column_exists("documents", "sync_attempts", inspector):
        op.drop_column("documents", "sync_attempts", schema=schema)

    if column_exists("documents", "last_sync_at", inspector):
        op.drop_column("documents", "last_sync_at", schema=schema)

    if column_exists("documents", "sync_state", inspector):
        # Drop composite index first
        op.drop_index(
            "ix_documents_sync_state_last_sync_at",
            table_name="documents",
            schema=schema,
        )
        op.drop_index("ix_documents_sync_state", table_name="documents", schema=schema)
        op.drop_column("documents", "sync_state", schema=schema)

    # Remove deleted_at column and index from documents
    if column_exists("documents", "deleted_at", inspector):
        op.drop_index("ix_documents_deleted_at", table_name="documents", schema=schema)
        op.drop_column("documents", "deleted_at", schema=schema)

    # NOTE: This downgrade does NOT restore the NOT NULL constraint on embedding columns
    # in message_embeddings and documents tables. This is intentional to avoid migration
    # failures if NULL embedding values exist (which is expected when using external vector stores).
