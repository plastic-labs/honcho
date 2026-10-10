"""Add observed work-unit attempt progress.

Revision ID: a7c2e4f819d0
Revises: 3e7a1c9d5b20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from src.config import settings

revision: str = "a7c2e4f819d0"
down_revision: str | None = "3e7a1c9d5b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    schema = settings.DB.SCHEMA
    op.add_column(
        "active_queue_sessions",
        sa.Column("current_queue_item_id", sa.BigInteger(), nullable=True),
        schema=schema,
    )
    # Do not backfill old claims: last_updated is a lease renewal, not a start.
    for name in ("started_at", "last_activity_at"):
        op.add_column(
            "active_queue_sessions",
            sa.Column(name, sa.DateTime(timezone=True), nullable=True),
            schema=schema,
        )
        op.alter_column(
            "active_queue_sessions", name, server_default=sa.func.now(), schema=schema
        )
    op.add_column(
        "active_queue_sessions",
        sa.Column("stage", sa.Text(), nullable=True),
        schema=schema,
    )
    op.alter_column(
        "active_queue_sessions", "stage", server_default="claimed", schema=schema
    )


def downgrade() -> None:
    for name in ("stage", "last_activity_at", "started_at", "current_queue_item_id"):
        op.drop_column("active_queue_sessions", name, schema=settings.DB.SCHEMA)
