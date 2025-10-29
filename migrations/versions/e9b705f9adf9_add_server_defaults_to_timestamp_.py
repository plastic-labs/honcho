"""add server defaults to timestamp boolean and jsonb columns

Revision ID: e9b705f9adf9
Revises: bb6fb3a7a643
Create Date: 2025-10-29 12:08:36.803611

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "e9b705f9adf9"
down_revision: str | None = "bb6fb3a7a643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


schema = get_schema()


def upgrade() -> None:
    # Add server defaults for timestamp columns
    op.alter_column(
        "workspaces",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "peers",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "messages",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "message_embeddings",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "collections",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "documents",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "queue",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "webhook_endpoints",
        "created_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "joined_at",
        server_default=sa.func.now(),
        schema=schema,
    )
    op.alter_column(
        "active_queue_sessions",
        "last_updated",
        server_default=sa.func.now(),
        schema=schema,
    )

    # Add server defaults for JSONB columns
    op.alter_column(
        "workspaces",
        "metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "workspaces",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "workspaces",
        "configuration",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "peers",
        "metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "peers",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "peers",
        "configuration",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "configuration",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "messages",
        "metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "messages",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "collections",
        "metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "collections",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "documents",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "configuration",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "internal_metadata",
        server_default=sa.text("'{}'::jsonb"),
        schema=schema,
    )

    # Add server defaults for boolean columns
    op.alter_column(
        "sessions",
        "is_active",
        server_default=sa.text("true"),
        schema=schema,
    )
    op.alter_column(
        "queue",
        "processed",
        server_default=sa.text("false"),
        schema=schema,
    )


def downgrade() -> None:
    # Remove server defaults for timestamp columns
    op.alter_column(
        "workspaces",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "peers",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "messages",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "message_embeddings",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "collections",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "documents",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "queue",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "webhook_endpoints",
        "created_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "joined_at",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "active_queue_sessions",
        "last_updated",
        server_default=None,
        schema=schema,
    )

    # Remove server defaults for JSONB columns
    op.alter_column(
        "workspaces",
        "metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "workspaces",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "workspaces",
        "configuration",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "peers",
        "metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "peers",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "peers",
        "configuration",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "sessions",
        "configuration",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "messages",
        "metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "messages",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "collections",
        "metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "collections",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "documents",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "configuration",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "session_peers",
        "internal_metadata",
        server_default=None,
        schema=schema,
    )

    # Remove server defaults for boolean columns
    op.alter_column(
        "sessions",
        "is_active",
        server_default=None,
        schema=schema,
    )
    op.alter_column(
        "queue",
        "processed",
        server_default=None,
        schema=schema,
    )
