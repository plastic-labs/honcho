"""make embeddings nullable, add soft delete, add vector sync state

This migration:
1. Makes embedding columns nullable in both message_embeddings and documents tables
   since embeddings are now stored in external vector stores instead of PostgreSQL.
2. Adds deleted_at column to documents table for soft delete support, enabling
   hybrid sync/soft delete pattern for vector store consistency.
3. Adds sync_state, last_sync_at, and sync_attempts columns to documents and
   message_embeddings tables for tracking vector store synchronization status.
4. Adds partial unique index on queue table for reconciler task deduplication,
   ensuring only one pending reconciler task exists per work_unit_key.
5. Makes workspace_name nullable on queue table for system-level tasks (e.g., reconciler)
   that don't belong to any specific workspace.

Revision ID: 119a52b73c60
Revises: 7c0d9a4e3b1f
Create Date: 2025-11-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import column_exists, constraint_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "119a52b73c60"
down_revision: str | None = "7c0d9a4e3b1f"
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
                server_default=sa.text(
                    "'pending'"
                ),  # Existing records need reconciliation
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
                server_default=sa.text("0"),
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
                server_default=sa.text(
                    "'pending'"
                ),  # Existing records need reconciliation
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
                server_default=sa.text("0"),
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

    # Add partial unique index on queue table for reconciler task deduplication
    # This ensures only one pending reconciler task exists per work_unit_key
    if not index_exists("queue", "uq_queue_work_unit_key", inspector):
        op.create_index(
            "uq_queue_work_unit_key",
            "queue",
            ["work_unit_key"],
            unique=True,
            schema=schema,
            postgresql_where=sa.text("task_type = 'reconciler' AND processed = false"),
        )

    # Make workspace_name nullable on queue table for system-level tasks
    # This requires dropping and recreating the FK constraint
    if constraint_exists("queue", "fk_queue_workspace_name", "foreignkey", inspector):
        op.drop_constraint(
            "fk_queue_workspace_name", "queue", type_="foreignkey", schema=schema
        )
    op.alter_column(
        "queue",
        "workspace_name",
        existing_type=sa.TEXT(),
        nullable=True,
        schema=schema,
    )
    op.create_foreign_key(
        "fk_queue_workspace_name",
        "queue",
        "workspaces",
        ["workspace_name"],
        ["name"],
        source_schema=schema,
        referent_schema=schema,
    )


def downgrade() -> None:
    """Remove deleted_at columns and revert embedding columns."""
    inspector = sa.inspect(op.get_bind())
    conn = op.get_bind()

    # Delete system-level queue items (with NULL workspace_name) before reverting to NOT NULL
    # First delete any active_queue_sessions referencing these queue items
    batch_size = 5000

    # Delete active_queue_sessions for queue items with NULL workspace_name
    while True:
        result = conn.execute(
            sa.text(
                f"""
                DELETE FROM "{schema}".active_queue_sessions
                WHERE work_unit_key IN (
                    SELECT work_unit_key FROM "{schema}".queue
                    WHERE workspace_name IS NULL
                )
                LIMIT :batch_size
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Delete queue items with NULL workspace_name
    while True:
        result = conn.execute(
            sa.text(
                f"""
                DELETE FROM "{schema}".queue
                WHERE workspace_name IS NULL
                LIMIT :batch_size
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Revert workspace_name to NOT NULL on queue table
    op.drop_constraint(
        "fk_queue_workspace_name", "queue", type_="foreignkey", schema=schema
    )
    op.alter_column(
        "queue",
        "workspace_name",
        existing_type=sa.TEXT(),
        nullable=False,
        schema=schema,
    )
    op.create_foreign_key(
        "fk_queue_workspace_name",
        "queue",
        "workspaces",
        ["workspace_name"],
        ["name"],
        source_schema=schema,
        referent_schema=schema,
    )

    # Drop reconciler queue index if it exists
    if index_exists("queue", "uq_queue_work_unit_key", inspector):
        op.drop_index(
            "uq_queue_work_unit_key",
            table_name="queue",
            schema=schema,
        )

    # Drop message_embeddings indexes if they exist
    if index_exists(
        "message_embeddings", "ix_message_embeddings_sync_state_last_sync_at", inspector
    ):
        op.drop_index(
            "ix_message_embeddings_sync_state_last_sync_at",
            table_name="message_embeddings",
            schema=schema,
        )
    if index_exists(
        "message_embeddings", "ix_message_embeddings_sync_state", inspector
    ):
        op.drop_index(
            "ix_message_embeddings_sync_state",
            table_name="message_embeddings",
            schema=schema,
        )

    if column_exists("message_embeddings", "sync_state", inspector):
        op.drop_column("message_embeddings", "sync_state", schema=schema)

    if column_exists("message_embeddings", "sync_attempts", inspector):
        op.drop_column("message_embeddings", "sync_attempts", schema=schema)

    if column_exists("message_embeddings", "last_sync_at", inspector):
        op.drop_column("message_embeddings", "last_sync_at", schema=schema)

    # Drop documents indexes if they exist
    if index_exists("documents", "ix_documents_sync_state_last_sync_at", inspector):
        op.drop_index(
            "ix_documents_sync_state_last_sync_at",
            table_name="documents",
            schema=schema,
        )
    if index_exists("documents", "ix_documents_sync_state", inspector):
        op.drop_index("ix_documents_sync_state", table_name="documents", schema=schema)

    if column_exists("documents", "sync_state", inspector):
        op.drop_column("documents", "sync_state", schema=schema)

    # Remove sync state columns from documents
    if column_exists("documents", "sync_attempts", inspector):
        op.drop_column("documents", "sync_attempts", schema=schema)

    if column_exists("documents", "last_sync_at", inspector):
        op.drop_column("documents", "last_sync_at", schema=schema)

    # Remove deleted_at column and index from documents
    if index_exists("documents", "ix_documents_deleted_at", inspector):
        op.drop_index("ix_documents_deleted_at", table_name="documents", schema=schema)
    if column_exists("documents", "deleted_at", inspector):
        op.drop_column("documents", "deleted_at", schema=schema)

    # NOTE: This downgrade does NOT restore the NOT NULL constraint on embedding columns
    # in message_embeddings and documents tables. This is intentional to avoid migration
    # failures if NULL embedding values exist (which is expected when using external vector stores).
