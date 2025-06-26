"""add messageembedding table

Revision ID: 917195d9b5e9
Revises: d429de0e5338
Create Date: 2024-01-01 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import index_exists, table_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "917195d9b5e9"
down_revision: str | None = "d429de0e5338"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
schema = settings.DB.SCHEMA


def upgrade() -> None:
    op.create_table(
        "message_embeddings",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("workspace_name", sa.Text(), nullable=False),
        sa.Column("session_name", sa.Text(), nullable=True),
        sa.Column("peer_name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Foreign key constraints
        sa.ForeignKeyConstraint(["message_id"], ["messages.public_id"]),
        sa.ForeignKeyConstraint(["workspace_name"], ["workspaces.name"]),
        sa.ForeignKeyConstraint(
            ["session_name", "workspace_name"],
            ["sessions.name", "sessions.workspace_name"],
        ),
        sa.ForeignKeyConstraint(
            ["peer_name", "workspace_name"], ["peers.name", "peers.workspace_name"]
        ),
        schema=schema,
    )

    # Create indexes
    op.create_index(
        "idx_message_embeddings_message_id",
        "message_embeddings",
        ["message_id"],
        schema=schema,
    )
    op.create_index(
        "idx_message_embeddings_workspace_name",
        "message_embeddings",
        ["workspace_name"],
        schema=schema,
    )
    op.create_index(
        "idx_message_embeddings_session_name",
        "message_embeddings",
        ["session_name"],
        schema=schema,
    )
    op.create_index(
        "idx_message_embeddings_peer_name",
        "message_embeddings",
        ["peer_name"],
        schema=schema,
    )
    op.create_index(
        "idx_message_embeddings_created_at",
        "message_embeddings",
        ["created_at"],
        schema=schema,
    )

    # Create HNSW index for vector similarity search
    op.execute(f"""
        CREATE INDEX idx_message_embeddings_embedding_hnsw
        ON {schema}.message_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not table_exists("message_embeddings", inspector):
        return

    # Drop indexes defensively
    indexes_to_drop = [
        "idx_message_embeddings_embedding_hnsw",
        "idx_message_embeddings_message_id",
        "idx_message_embeddings_workspace_name",
        "idx_message_embeddings_session_name",
        "idx_message_embeddings_peer_name",
        "idx_message_embeddings_created_at",
    ]

    for index_name in indexes_to_drop:
        if index_exists("message_embeddings", index_name, inspector):
            op.drop_index(index_name, table_name="message_embeddings", schema=schema)

    # Drop table (this will also drop foreign keys and check constraints)
    op.drop_table("message_embeddings", schema=schema)
