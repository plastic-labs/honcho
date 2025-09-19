"""Add token_count to QueueItem table

Revision ID: 394e7c39362d
Revises: 88b0fb10906f
Create Date: 2025-08-27 10:49:26.591473

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "394e7c39362d"
down_revision: str | None = "88b0fb10906f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    op.add_column(
        "queue",
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        schema=schema,
    )

    # ### Data Migration: Backfill token_count using raw SQL ###
    bind = op.get_bind()
    schema_name = settings.DB.SCHEMA

    BATCH_SIZE = 500  # Process 500 items at a time

    while True:
        # Fetch a batch of items that need backfilling using raw SQL
        items_result = bind.execute(
            sa.text(
                f"""
                SELECT id, payload FROM {schema_name}.queue
                WHERE processed = false
                  AND token_count = 0
                  AND task_type IN ('representation', 'summary')
                ORDER BY id
                LIMIT :batch_size
                """
            ),
            {"batch_size": BATCH_SIZE},
        )
        items_to_backfill = items_result.fetchall()

        if not items_to_backfill:
            break  # No more items to process

        # Assuming message_id is always present due to application-level validation
        message_id_map = {
            item.id: item.payload["message_id"] for item in items_to_backfill
        }

        # Get token counts for the message IDs
        token_counts_result = bind.execute(
            sa.text(
                f"""
                SELECT id, token_count FROM {schema_name}.messages
                WHERE id = ANY(:ids)
                """
            ),
            {"ids": list(message_id_map.values())},
        )
        token_map = {msg_id: count for msg_id, count in token_counts_result.fetchall()}

        # Prepare parameters for the bulk update, skipping any queue items whose
        # message_id was not found in the messages table.
        update_params = [
            (qid, token_map.get(mid))
            for qid, mid in message_id_map.items()
            if token_map.get(mid) is not None
        ]

        if update_params:
            queue_ids = [p[0] for p in update_params]
            token_counts = [p[1] for p in update_params]

            # Perform a single bulk update using the UNNEST pattern
            bind.execute(
                sa.text(
                    f"""
                    UPDATE {schema_name}.queue q
                    SET token_count = v.token_count
                    FROM (
                        SELECT
                            UNNEST(:queue_ids) AS queue_id,
                            UNNEST(:token_counts) AS token_count
                    ) AS v
                    WHERE q.id = v.queue_id
                    """
                ),
                {"queue_ids": queue_ids, "token_counts": token_counts},
            )

        # If we fetched fewer items than the batch size, we are on the last batch
        if len(items_to_backfill) < BATCH_SIZE:
            break


def downgrade() -> None:
    inspector = sa.inspect(op.get_context().connection)
    if column_exists("queue", "token_count", inspector):
        op.drop_column("queue", "token_count", schema=schema)
