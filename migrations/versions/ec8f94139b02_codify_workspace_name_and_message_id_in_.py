"""codify workspace_name and message_id in queue table

Revision ID: ec8f94139b02
Revises: e9b705f9adf9
Create Date: 2025-10-28 17:39:51.778665

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, fk_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "ec8f94139b02"
down_revision: str | None = "e9b705f9adf9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


schema = get_schema()


def upgrade() -> None:
    # Step 1: Drop rows where workspace_name is NULL in payload
    # This removes invalid/corrupted queue items
    conn = op.get_bind()
    batch_size = 10000
    while True:
        result = conn.execute(
            sa.text(
                f"""
                DELETE FROM "{schema}".queue
                WHERE id IN (
                    SELECT id FROM "{schema}".queue
                    WHERE payload->>'workspace_name' IS NULL
                    LIMIT :batch_size
                )
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 2: Add workspace_name column (nullable initially for backfill)
    op.add_column(
        "queue",
        sa.Column("workspace_name", sa.TEXT(), nullable=True),
        schema=schema,
    )

    # Step 3: Add message_id column (nullable, as not all tasks have message_id)
    op.add_column(
        "queue",
        sa.Column("message_id", sa.BigInteger(), nullable=True),
        schema=schema,
    )

    # Step 4: Backfill workspace_name and message_id from payload in batches
    batch_size = 5000
    while True:
        result = conn.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM "{schema}".queue
                    WHERE workspace_name IS NULL
                       OR (message_id IS NULL AND payload ? 'message_id' AND payload->>'message_id' IS NOT NULL)
                    LIMIT :batch_size
                )
                UPDATE "{schema}".queue q
                SET
                    workspace_name = COALESCE(q.workspace_name, q.payload->>'workspace_name'),
                    message_id = COALESCE(
                        q.message_id,
                        CASE
                            WHEN q.payload ? 'message_id' AND q.payload->>'message_id' IS NOT NULL
                            THEN (q.payload->>'message_id')::bigint
                        END
                    )
                FROM batch
                WHERE q.id = batch.id
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 5: Remove workspace_name and message_id from JSONB payloads in batches
    while True:
        result = conn.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM "{schema}".queue
                    WHERE payload ? 'workspace_name' OR payload ? 'message_id'
                    LIMIT :batch_size
                )
                UPDATE "{schema}".queue q
                SET payload = q.payload - 'workspace_name' - 'message_id'
                FROM batch
                WHERE q.id = batch.id
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 6: Make workspace_name non-nullable
    op.alter_column("queue", "workspace_name", nullable=False, schema=schema)

    # Step 7: Add foreign key constraint on workspace_name -> workspaces.name
    op.create_foreign_key(
        "fk_queue_workspace_name",
        "queue",
        "workspaces",
        ["workspace_name"],
        ["name"],
        source_schema=schema,
        referent_schema=schema,
    )

    # Step 7b: Add foreign key constraint on message_id -> messages.id
    op.create_foreign_key(
        "fk_queue_message_id",
        "queue",
        "messages",
        ["message_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
    )

    # Step 8: Add index on workspace_name (for FK performance and filtering)
    op.create_index(
        op.f("ix_queue_workspace_name"),
        "queue",
        ["workspace_name"],
        unique=False,
        schema=schema,
    )

    # Step 9: Add partial index on message_id WHERE message_id IS NOT NULL
    # This optimizes JOINs with the messages table
    op.create_index(
        "ix_queue_message_id_not_null",
        "queue",
        ["message_id"],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("message_id IS NOT NULL"),
    )

    # Step 10: Add composite index on (workspace_name, processed)
    # This optimizes queries that filter unprocessed items by workspace
    op.create_index(
        "ix_queue_workspace_name_processed",
        "queue",
        ["workspace_name", "processed"],
        unique=False,
        schema=schema,
    )

    # Step 11: Add composite index on (work_unit_key, processed, id)
    # This is critical for the hot path: "get next unprocessed item for this work unit"
    # Covers: WHERE work_unit_key = ? AND NOT processed ORDER BY id
    op.create_index(
        "ix_queue_work_unit_key_processed_id",
        "queue",
        ["work_unit_key", "processed", "id"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    # Drop indexes
    if index_exists("queue", "ix_queue_work_unit_key_processed_id", inspector):
        op.drop_index(
            "ix_queue_work_unit_key_processed_id", table_name="queue", schema=schema
        )

    if index_exists("queue", "ix_queue_workspace_name_processed", inspector):
        op.drop_index(
            "ix_queue_workspace_name_processed", table_name="queue", schema=schema
        )

    if index_exists("queue", "ix_queue_message_id_not_null", inspector):
        op.drop_index("ix_queue_message_id_not_null", table_name="queue", schema=schema)

    if index_exists("queue", "ix_queue_workspace_name", inspector):
        op.drop_index(
            op.f("ix_queue_workspace_name"), table_name="queue", schema=schema
        )

    # Drop foreign key constraints
    if fk_exists("queue", "fk_queue_message_id", inspector):
        op.drop_constraint("fk_queue_message_id", "queue", schema=schema)

    if fk_exists("queue", "fk_queue_workspace_name", inspector):
        op.drop_constraint("fk_queue_workspace_name", "queue", schema=schema)

    # Restore workspace_name and message_id to payload in batches
    conn = op.get_bind()
    batch_size = 5000

    if column_exists("queue", "workspace_name", inspector) or column_exists(
        "queue", "message_id", inspector
    ):
        while True:
            result = conn.execute(
                sa.text(
                    f"""
                    WITH batch AS (
                        SELECT id
                        FROM "{schema}".queue
                        WHERE (workspace_name IS NOT NULL AND NOT (payload ? 'workspace_name'))
                           OR (message_id IS NOT NULL AND NOT (payload ? 'message_id'))
                        LIMIT :batch_size
                    )
                    UPDATE "{schema}".queue q
                    SET payload = q.payload
                        || CASE WHEN q.workspace_name IS NOT NULL AND NOT (q.payload ? 'workspace_name')
                                THEN jsonb_build_object('workspace_name', q.workspace_name)
                                ELSE '{{}}'::jsonb END
                        || CASE WHEN q.message_id IS NOT NULL AND NOT (q.payload ? 'message_id')
                                THEN jsonb_build_object('message_id', q.message_id)
                                ELSE '{{}}'::jsonb END
                    FROM batch
                    WHERE q.id = batch.id
                    """
                ),
                {"batch_size": batch_size},
            )
            if result.rowcount == 0:
                break

    # Drop columns
    if column_exists("queue", "message_id", inspector):
        op.drop_column("queue", "message_id", schema=schema)

    if column_exists("queue", "workspace_name", inspector):
        op.drop_column("queue", "workspace_name", schema=schema)
