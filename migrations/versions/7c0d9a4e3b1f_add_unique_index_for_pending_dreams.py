"""add unique index for pending dream queue deduplication

This ensures `enqueue_dream()` is idempotent under concurrent calls by enforcing
at most one unprocessed dream queue item per `work_unit_key`.

Revision ID: 7c0d9a4e3b1f
Revises: f1a2b3c4d5e6
Create Date: 2026-01-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "7c0d9a4e3b1f"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add a partial unique index to prevent duplicate pending dream queue items."""
    op.create_index(
        "uq_queue_dream_pending_work_unit_key",
        "queue",
        ["work_unit_key"],
        unique=True,
        schema=schema,
        postgresql_where=sa.text("task_type = 'dream' AND processed = false"),
    )


def downgrade() -> None:
    """Drop the partial unique index for pending dream queue items."""
    op.drop_index(
        "uq_queue_dream_pending_work_unit_key", table_name="queue", schema=schema
    )
