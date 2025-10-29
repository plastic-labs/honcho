"""codify workspace_name and message_id in queue table

Revision ID: ec8f94139b02
Revises: bb6fb3a7a643
Create Date: 2025-10-28 17:39:51.778665

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, fk_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "ec8f94139b02"
down_revision: str | None = "bb6fb3a7a643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


schema = get_schema()


def upgrade() -> None:
    # Step 1: Drop rows where workspace_name is NULL in payload
    # This removes invalid/corrupted queue items
    op.execute(
        sa.text(
            f"""
            DELETE FROM "{schema}".queue
            WHERE payload->>'workspace_name' IS NULL
            """
        )
    )

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

    # Step 4: Backfill workspace_name from payload in batches
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                rows_updated INT;
            BEGIN
                LOOP
                    UPDATE "{schema}".queue
                    SET workspace_name = payload->>'workspace_name'
                    WHERE id IN (
                        SELECT id FROM "{schema}".queue
                        WHERE workspace_name IS NULL
                        LIMIT 1000
                    );

                    GET DIAGNOSTICS rows_updated = ROW_COUNT;
                    EXIT WHEN rows_updated = 0;
                END LOOP;
            END $$;
            """
        )
    )

    # Step 5: Backfill message_id from payload in batches (only where it exists)
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                rows_updated INT;
            BEGIN
                LOOP
                    UPDATE "{schema}".queue
                    SET message_id = (payload->>'message_id')::bigint
                    WHERE id IN (
                        SELECT id FROM "{schema}".queue
                        WHERE message_id IS NULL
                          AND payload ? 'message_id'
                          AND payload->>'message_id' IS NOT NULL
                        LIMIT 1000
                    );

                    GET DIAGNOSTICS rows_updated = ROW_COUNT;
                    EXIT WHEN rows_updated = 0;
                END LOOP;
            END $$;
            """
        )
    )

    # Step 6: Remove workspace_name from JSONB payloads in batches
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                rows_updated INT;
            BEGIN
                LOOP
                    UPDATE "{schema}".queue
                    SET payload = payload - 'workspace_name'
                    WHERE id IN (
                        SELECT id FROM "{schema}".queue
                        WHERE payload ? 'workspace_name'
                        LIMIT 1000
                    );

                    GET DIAGNOSTICS rows_updated = ROW_COUNT;
                    EXIT WHEN rows_updated = 0;
                END LOOP;
            END $$;
            """
        )
    )

    # Step 7: Remove message_id from JSONB payloads in batches
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                rows_updated INT;
            BEGIN
                LOOP
                    UPDATE "{schema}".queue
                    SET payload = payload - 'message_id'
                    WHERE id IN (
                        SELECT id FROM "{schema}".queue
                        WHERE payload ? 'message_id'
                        LIMIT 1000
                    );

                    GET DIAGNOSTICS rows_updated = ROW_COUNT;
                    EXIT WHEN rows_updated = 0;
                END LOOP;
            END $$;
            """
        )
    )

    # Step 8: Make workspace_name non-nullable
    op.alter_column("queue", "workspace_name", nullable=False, schema=schema)

    # Step 9: Add foreign key constraint on workspace_name -> workspaces.name
    op.create_foreign_key(
        "fk_queue_workspace_name",
        "queue",
        "workspaces",
        ["workspace_name"],
        ["name"],
        source_schema=schema,
        referent_schema=schema,
    )

    # Step 10: Add index on workspace_name (for FK performance and filtering)
    op.create_index(
        op.f("ix_queue_workspace_name"),
        "queue",
        ["workspace_name"],
        unique=False,
        schema=schema,
    )

    # Step 11: Add partial index on message_id WHERE message_id IS NOT NULL
    # This optimizes JOINs with the messages table
    op.create_index(
        "ix_queue_message_id_not_null",
        "queue",
        ["message_id"],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("message_id IS NOT NULL"),
    )

    # Step 12: Add composite index on (workspace_name, processed)
    # This optimizes queries that filter unprocessed items by workspace
    op.create_index(
        "ix_queue_workspace_name_processed",
        "queue",
        ["workspace_name", "processed"],
        unique=False,
        schema=schema,
    )

    # Step 13: Add composite index on (work_unit_key, processed, id)
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

    # Drop foreign key constraint
    if fk_exists("queue", "fk_queue_workspace_name", inspector):
        op.drop_constraint("fk_queue_workspace_name", "queue", schema=schema)

    # Restore workspace_name and message_id to payload in batches
    if column_exists("queue", "workspace_name", inspector):
        op.execute(
            sa.text(
                f"""
                DO $$
                DECLARE
                    rows_updated INT;
                BEGIN
                    LOOP
                        UPDATE "{schema}".queue
                        SET payload = jsonb_set(payload, '{{workspace_name}}', to_jsonb(workspace_name))
                        WHERE id IN (
                            SELECT id FROM "{schema}".queue
                            WHERE workspace_name IS NOT NULL
                              AND NOT (payload ? 'workspace_name')
                            LIMIT 1000
                        );

                        GET DIAGNOSTICS rows_updated = ROW_COUNT;
                        EXIT WHEN rows_updated = 0;
                    END LOOP;
                END $$;
                """
            )
        )

    if column_exists("queue", "message_id", inspector):
        op.execute(
            sa.text(
                f"""
                DO $$
                DECLARE
                    rows_updated INT;
                BEGIN
                    LOOP
                        UPDATE "{schema}".queue
                        SET payload = jsonb_set(payload, '{{message_id}}', to_jsonb(message_id))
                        WHERE id IN (
                            SELECT id FROM "{schema}".queue
                            WHERE message_id IS NOT NULL
                              AND NOT (payload ? 'message_id')
                            LIMIT 1000
                        );

                        GET DIAGNOSTICS rows_updated = ROW_COUNT;
                        EXIT WHEN rows_updated = 0;
                    END LOOP;
                END $$;
                """
            )
        )

    # Drop columns
    if column_exists("queue", "message_id", inspector):
        op.drop_column("queue", "message_id", schema=schema)

    if column_exists("queue", "workspace_name", inspector):
        op.drop_column("queue", "workspace_name", schema=schema)
