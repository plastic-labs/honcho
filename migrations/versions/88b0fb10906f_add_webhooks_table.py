"""add webhooks table

Revision ID: 88b0fb10906f
Revises: 05486ce795d5
Create Date: 2025-07-25 16:12:11.015327

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "88b0fb10906f"
down_revision: str | None = "05486ce795d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    op.create_table(
        "webhooks",
        sa.Column("id", sa.TEXT(), nullable=False),
        sa.Column("workspace_name", sa.TEXT(), nullable=False),
        sa.Column("url", sa.TEXT(), nullable=False),
        sa.Column("event", sa.TEXT(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("secret", sa.TEXT(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_name"],
            ["workspaces.name"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_name",
            "url",
            "event",
            name="unique_webhook_url_event_per_workspace",
        ),
        sa.CheckConstraint("length(id) = 21", name="webhook_id_length"),
        sa.CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="webhook_id_format"),
        sa.CheckConstraint("length(url) <= 2048", name="webhook_url_length"),
    )
    op.create_index(
        op.f("ix_webhooks_event"),
        "webhooks",
        ["event"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        op.f("ix_webhooks_workspace_name"),
        "webhooks",
        ["workspace_name"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        op.f("ix_webhooks_created_at"),
        "webhooks",
        ["created_at"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    if index_exists("webhooks", "ix_webhooks_created_at", schema):
        op.drop_index(
            op.f("ix_webhooks_created_at"), table_name="webhooks", schema=schema
        )
    if index_exists("webhooks", "ix_webhooks_workspace_name", schema):
        op.drop_index(
            op.f("ix_webhooks_workspace_name"), table_name="webhooks", schema=schema
        )
    if index_exists("webhooks", "ix_webhooks_event", schema):
        op.drop_index(op.f("ix_webhooks_event"), table_name="webhooks", schema=schema)
    op.drop_table("webhooks", schema=schema)
