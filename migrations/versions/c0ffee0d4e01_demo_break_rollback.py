"""DEMO, DO NOT MERGE: two changes that break rollback to the previous image

1. messages.content_hash is added NOT NULL with no default. The candidate
   fills it from Python; the previous image does not know the column, so
   every message it writes fails.
2. documents.provenance is a perfectly legal nullable expand column. The
   candidate writes conclusion premises there instead of document_sources,
   so the previous image reads every new deduction as having no premises,
   the way a swap back after plastic-labs/honcho#952 would have lost them.

Revision ID: c0ffee0d4e01
Revises: 3e7a1c9d5b20
Create Date: 2026-10-16
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from migrations.utils import get_schema

revision = "c0ffee0d4e01"
down_revision = "3e7a1c9d5b20"
branch_labels = None
depends_on = None

schema = get_schema()


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("content_hash", sa.TEXT(), nullable=False, server_default=""),
        schema=schema,
    )
    op.alter_column("messages", "content_hash", server_default=None, schema=schema)
    op.add_column(
        "documents", sa.Column("provenance", JSONB(), nullable=True), schema=schema
    )


def downgrade() -> None:
    op.drop_column("documents", "provenance", schema=schema)
    op.drop_column("messages", "content_hash", schema=schema)
