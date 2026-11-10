"""add webhook_endpoints.event_types

Revision ID: 0e32bf9d5a22
Revises: 4f9c2a7e8d31
Create Date: 2026-11-10

The event types an endpoint receives. NULL receives every type, which is how
every endpoint behaves before this revision, so an upgraded deployment delivers
exactly as before until an endpoint sets a list. ADD COLUMN with no default is a
catalog-only change, on the partitioned parent and on each partition. An image
from before this revision never reads or writes the column, so it can still run
against it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from migrations.utils import column_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "0e32bf9d5a22"
down_revision: str | None = "4f9c2a7e8d31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    if not column_exists("webhook_endpoints", "event_types"):
        op.add_column(
            "webhook_endpoints",
            sa.Column("event_types", postgresql.ARRAY(sa.TEXT()), nullable=True),
            schema=schema,
        )


def downgrade() -> None:
    if column_exists("webhook_endpoints", "event_types"):
        op.drop_column("webhook_endpoints", "event_types", schema=schema)
