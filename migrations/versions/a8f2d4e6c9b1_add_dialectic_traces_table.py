"""add dialectic_traces table

Revision ID: a8f2d4e6c9b1
Revises: e4eba9cfaa6f
Create Date: 2026-01-29

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "a8f2d4e6c9b1"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Create the dialectic_traces table for internal logging of dialectic interactions."""
    op.create_table(
        "dialectic_traces",
        sa.Column("id", sa.TEXT(), nullable=False),
        sa.Column("workspace_name", sa.TEXT(), nullable=False),
        sa.Column("session_name", sa.TEXT(), nullable=True),
        sa.Column("observer", sa.TEXT(), nullable=False),
        sa.Column("observed", sa.TEXT(), nullable=False),
        sa.Column("query", sa.TEXT(), nullable=False),
        sa.Column(
            "retrieved_doc_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("response", sa.TEXT(), nullable=False),
        sa.Column("reasoning_level", sa.TEXT(), nullable=False),
        sa.Column("total_duration_ms", sa.Float(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["workspace_name"],
            [f"{schema}.workspaces.name" if schema else "workspaces.name"],
            name="fk_dialectic_traces_workspace_name",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=schema,
    )
    # Create indexes for efficient querying
    op.create_index(
        "ix_dialectic_traces_workspace_name",
        "dialectic_traces",
        ["workspace_name"],
        schema=schema,
    )
    op.create_index(
        "ix_dialectic_traces_session_name",
        "dialectic_traces",
        ["session_name"],
        schema=schema,
    )
    op.create_index(
        "ix_dialectic_traces_observer",
        "dialectic_traces",
        ["observer"],
        schema=schema,
    )
    op.create_index(
        "ix_dialectic_traces_observed",
        "dialectic_traces",
        ["observed"],
        schema=schema,
    )
    op.create_index(
        "ix_dialectic_traces_created_at",
        "dialectic_traces",
        ["created_at"],
        schema=schema,
    )


def downgrade() -> None:
    """Drop the dialectic_traces table."""
    op.drop_index(
        "ix_dialectic_traces_created_at", table_name="dialectic_traces", schema=schema
    )
    op.drop_index(
        "ix_dialectic_traces_observed", table_name="dialectic_traces", schema=schema
    )
    op.drop_index(
        "ix_dialectic_traces_observer", table_name="dialectic_traces", schema=schema
    )
    op.drop_index(
        "ix_dialectic_traces_session_name", table_name="dialectic_traces", schema=schema
    )
    op.drop_index(
        "ix_dialectic_traces_workspace_name",
        table_name="dialectic_traces",
        schema=schema,
    )
    op.drop_table("dialectic_traces", schema=schema)
