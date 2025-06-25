"""add embedding column to Messages table

Revision ID: 48b78978df12
Revises: d429de0e5338
Create Date: 2025-06-25 10:23:39.808307

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy import text

from migrations.utils import column_exists, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "48b78978df12"
down_revision: Union[str, None] = "d429de0e5338"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
schema = settings.DB.SCHEMA


def upgrade() -> None:
    op.add_column("messages", sa.Column("embedding", Vector(1536), nullable=True))
    try:
        op.execute(
            text(
                f"""
                CREATE INDEX idx_messages_embedding_hnsw ON {schema}.messages 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m=16, ef_construction=64);
                """
            )
        )
        print(
            f"HNSW index idx_messages_embedding_hnsw created on {schema}.messages table"
        )
    except Exception as e:
        print(
            f"Error creating HNSW index idx_messages_embedding_hnsw on {schema}.messages table: {e}"
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if index_exists("messages", "idx_messages_embedding_hnsw", inspector):
        op.drop_index("idx_messages_embedding_hnsw", "messages", schema=schema)
    if column_exists("messages", "embedding", inspector):
        op.drop_column("messages", "embedding", schema=schema)
