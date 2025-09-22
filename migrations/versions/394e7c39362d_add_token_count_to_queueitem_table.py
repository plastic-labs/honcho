"""Add token_count to QueueItem table

Revision ID: 394e7c39362d
Revises: 88b0fb10906f
Create Date: 2025-08-27 10:49:26.591473

"""

import json
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
        items_stmt = sa.text(
            f"""
            SELECT id, payload FROM {schema_name}.queue
            WHERE processed = false
              AND token_count = 0
              AND task_type IN ('representation', 'summary')
            ORDER BY id
            LIMIT :batch_size
            """
        ).columns(id=sa.Integer, payload=sa.JSON)

        items_to_backfill = (
            bind.execute(
                items_stmt,
                {"batch_size": BATCH_SIZE},
            )
            .mappings()
            .all()
        )

        if not items_to_backfill:
            break  # No more items to process

        # Assuming message_id is always present due to application-level validation
        def _extract_message_id(payload: object) -> int | None:
            if isinstance(payload, dict):
                return payload.get("message_id")  # type: ignore[return-value]
            if payload is None:
                return None
            try:
                parsed = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return None
            if isinstance(parsed, dict):
                return parsed.get("message_id")  # type: ignore[return-value]
            return None

        message_id_map = {
            row["id"]: _extract_message_id(row["payload"]) for row in items_to_backfill
        }
        # Drop rows where message_id couldn't be extracted
        message_id_map = {
            queue_id: message_id
            for queue_id, message_id in message_id_map.items()
            if message_id is not None
        }

        # Get token counts for the message IDs
        token_ids = list(message_id_map.values())
        if not token_ids:
            continue

        placeholders = ", ".join(f":id_{idx}" for idx in range(len(token_ids)))
        token_counts_stmt = sa.text(
            f"""
            SELECT id, token_count FROM {schema_name}.messages
            WHERE id IN ({placeholders})
            """
        )
        bind_params = {f"id_{idx}": token_id for idx, token_id in enumerate(token_ids)}
        token_counts_result = bind.execute(token_counts_stmt, bind_params)
        token_map = {msg_id: count for msg_id, count in token_counts_result.fetchall()}

        # Prepare parameters for the bulk update, skipping any queue items whose
        # message_id was not found in the messages table.
        update_params = [
            (qid, token_map.get(mid))
            for qid, mid in message_id_map.items()
            if token_map.get(mid) is not None
        ]

        if update_params:
            update_stmt = sa.text(
                f"UPDATE {schema_name}.queue SET token_count = :token_count WHERE id = :queue_id"
            )

            bind.execute(
                update_stmt,
                [
                    {"queue_id": queue_id, "token_count": token_count}
                    for queue_id, token_count in update_params
                ],
            )

        # If we fetched fewer items than the batch size, we are on the last batch
        if len(items_to_backfill) < BATCH_SIZE:
            break


def downgrade() -> None:
    inspector = sa.inspect(op.get_context().connection)
    if column_exists("queue", "token_count", inspector):
        op.drop_column("queue", "token_count", schema=schema)
