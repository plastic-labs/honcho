"""Add inactive primitive tombstone columns.

Expand only: nullable timestamps without defaults, backfills, or new indexes.
Old writers can omit these columns. This does not enable deletion or establish
an erasure-safe rollback floor: readers and writers must adopt the lifecycle
before any code starts setting these timestamps. Retain the existing document
indexes and queue discovery indexes for prior-image compatibility.

The migration runner supplies the advisory lock, lock timeout, and retries.

Revision ID: d8c4f6a2b901
Revises: a5f8da8f6d23
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema

revision: str = "d8c4f6a2b901"
down_revision: str | None = "a5f8da8f6d23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("workspaces", "peers", "sessions", "messages")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(
            table,
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            schema=get_schema(),
            if_not_exists=True,
        )


def downgrade() -> None:
    # Development reversal only; production rollback is an image swap.
    for table in reversed(TABLES):
        op.drop_column(table, "deleted_at", schema=get_schema(), if_exists=True)
