"""add error field to QueueItem

Revision ID: ddf7b217ca3e
Revises: 88b0fb10906f
Create Date: 2025-10-06 17:24:27.580533

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "ddf7b217ca3e"
down_revision: str | None = "88b0fb10906f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    # Add error column
    op.add_column(
        "queue",
        sa.Column("error", sa.TEXT(), nullable=True),
        schema=schema,
    )

    op.add_column(
        "queue",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.func.now(),
        ),
        schema=schema,
    )

    # Backfill created_at for existing queue items (batched)
    op.execute(
        sa.text(
            f"""
            DO $$
            DECLARE
                rows_updated INT;
            BEGIN
                LOOP
                    UPDATE {schema}.queue
                    SET created_at = NOW()
                    WHERE id IN (
                        SELECT id FROM {schema}.queue
                        WHERE created_at IS NULL
                        LIMIT 1000
                    );

                    GET DIAGNOSTICS rows_updated = ROW_COUNT;
                    EXIT WHEN rows_updated = 0;
                END LOOP;
            END $$;
            """
        )
    )

    # Make created_at non-nullable
    op.alter_column("queue", "created_at", nullable=False, schema=schema)

    # Add index on created_at
    op.create_index(
        op.f("ix_queue_created_at"),
        "queue",
        ["created_at"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if column_exists("queue", "error", inspector):
        op.drop_column("queue", "error", schema=schema)

    # Drop index first
    if index_exists("queue", "ix_queue_created_at", inspector):
        op.drop_index(op.f("ix_queue_created_at"), table_name="queue", schema=schema)

    if column_exists("queue", "created_at", inspector):
        op.drop_column("queue", "created_at", schema=schema)
