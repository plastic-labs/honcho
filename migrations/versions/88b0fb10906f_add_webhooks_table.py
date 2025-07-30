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
        "webhook_endpoints",
        sa.Column(
            "id",
            sa.TEXT(),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "workspace_name",
            sa.TEXT(),
            sa.ForeignKey("workspaces.name"),
            nullable=True,
        ),
        sa.Column("url", sa.TEXT(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("length(url) <= 2048", name="webhook_endpoint_url_length"),
        schema=schema,
    )

    op.create_index(
        op.f("idx_webhook_endpoints_workspace_lookup"),
        "webhook_endpoints",
        ["workspace_name"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if index_exists(
        "webhook_endpoints", "idx_webhook_endpoints_workspace_lookup", inspector
    ):
        op.drop_index(
            op.f("idx_webhook_endpoints_workspace_lookup"),
            table_name="webhook_endpoints",
            schema=schema,
        )

    # Drop webhook_endpoints table
    op.drop_table("webhook_endpoints", schema=schema)
