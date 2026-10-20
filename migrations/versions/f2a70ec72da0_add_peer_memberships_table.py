"""add peer_memberships table

One row per membership: member_id belongs to parent_id within a workspace. Both
sides reference peers by their stable id rather than by name, so renaming a
peer never touches this table. Reads aggregate across memberships at query
time, so this revision is DDL only: additive, on an empty table, with no
backfill.

Revision ID: f2a70ec72da0
Revises: a5f8da8f6d23
Create Date: 2026-10-20

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from migrations.utils import get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "f2a70ec72da0"
down_revision: str | None = "a5f8da8f6d23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if table_exists("peer_memberships", inspector):
        return

    op.create_table(
        "peer_memberships",
        sa.Column("id", sa.TEXT, nullable=False),
        sa.Column("workspace_name", sa.TEXT, nullable=False),
        sa.Column("parent_id", sa.TEXT, nullable=False),
        sa.Column("member_id", sa.TEXT, nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["workspace_name"],
            [f"{schema}.workspaces.name"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            [f"{schema}.peers.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["member_id"],
            [f"{schema}.peers.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "workspace_name",
            "parent_id",
            "member_id",
        ),
        sa.CheckConstraint("parent_id <> member_id", name="not_self"),
        sa.CheckConstraint("length(id) = 21", name="id_length"),
        sa.CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        schema=schema,
    )
    op.create_index(
        "ix_peer_memberships_parent",
        "peer_memberships",
        ["workspace_name", "parent_id"],
        schema=schema,
    )
    op.create_index(
        "ix_peer_memberships_member",
        "peer_memberships",
        ["workspace_name", "member_id"],
        schema=schema,
    )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if table_exists("peer_memberships", inspector):
        op.drop_table("peer_memberships", schema=schema)
