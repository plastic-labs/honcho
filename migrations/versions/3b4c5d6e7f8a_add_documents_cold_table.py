"""add documents_cold table for eviction cold storage

Revision ID: 3b4c5d6e7f8a
Revises: 2a3b4c5d6e7f
Create Date: 2026-06-23 19:00:00.000000

"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from alembic import op

revision: str = "3b4c5d6e7f8a"
down_revision: str | None = "2a3b4c5d6e7f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents_cold",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("workspace_name", sa.Text(), nullable=False),
        sa.Column("collection_name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("level", sa.Text(), nullable=True),
        sa.Column("metadata", JSONB(), server_default=sa.text("NULL"), nullable=True),
        sa.Column("internal_metadata", JSONB(), server_default=sa.text("NULL"), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("evicted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("edge_snapshot", JSONB(), server_default=sa.text("NULL"), nullable=True),
        sa.Column("access_log_tail", JSONB(), server_default=sa.text("NULL"), nullable=True),
        sa.Column("rehydrated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_cold_workspace", "documents_cold", ["workspace_name"])
    op.create_index("ix_documents_cold_evicted_at", "documents_cold", ["evicted_at"])


def downgrade() -> None:
    op.drop_table("documents_cold")
