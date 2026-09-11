"""add session last_message_at

Revision ID: cfaff339d519
Revises: e4eba9cfaa6f
Create Date: 2026-08-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "cfaff339d519"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()
INDEX_NAME = "ix_sessions_workspace_last_message_at"


def upgrade() -> None:
    """Add the nullable session activity timestamp."""
    op.add_column(
        "sessions",
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}"."sessions" AS session
            SET last_message_at = activity.last_message_at
            FROM (
                SELECT workspace_name, session_name, MAX(created_at) AS last_message_at
                FROM "{schema}"."messages"
                GROUP BY workspace_name, session_name
            ) AS activity
            WHERE session.workspace_name = activity.workspace_name
              AND session.name = activity.session_name
            """
        )
    )
    op.create_index(
        INDEX_NAME,
        "sessions",
        [
            "workspace_name",
            sa.text("last_message_at DESC NULLS LAST"),
            sa.text("id DESC"),
        ],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    """Remove the session activity timestamp."""
    op.drop_index(INDEX_NAME, table_name="sessions", schema=schema)
    op.drop_column("sessions", "last_message_at", schema=schema)
