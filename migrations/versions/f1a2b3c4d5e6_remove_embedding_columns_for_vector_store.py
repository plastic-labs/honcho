"""remove embedding columns for vector store migration

This migration removes the embedding columns from the message_embeddings and documents
tables as part of the migration from pgvector to external vector stores (turbopuffer/lancedb).

Revision ID: f1a2b3c4d5e6
Revises: baa22cad81e2
Create Date: 2025-11-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "baa22cad81e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Remove embedding columns and HNSW indexes from message_embeddings and documents tables.
    Also add chunk_index column to message_embeddings for tracking chunked message embeddings.
    """
    inspector = sa.inspect(op.get_bind())

    # === message_embeddings table ===

    # Drop HNSW index on message_embeddings.embedding if it exists
    # Check for both possible index names (old naming vs new naming convention)
    for index_name in [
        "ix_message_embeddings_embedding_hnsw",
        "idx_message_embeddings_embedding_hnsw",
    ]:
        if index_exists("message_embeddings", index_name, inspector):
            op.drop_index(index_name, table_name="message_embeddings", schema=schema)

    # Drop embedding column from message_embeddings if it exists
    if column_exists("message_embeddings", "embedding", inspector):
        op.drop_column("message_embeddings", "embedding", schema=schema)

    # Add chunk_index column to message_embeddings if it doesn't exist
    # This is needed to track which chunk of a message this embedding represents
    # Vector ID format: {message_public_id}_{chunk_index}
    if not column_exists("message_embeddings", "chunk_index", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "chunk_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )

    # === documents table ===

    # Drop HNSW index on documents.embedding if it exists
    # Check for both possible index names (old naming vs new naming convention)
    for index_name in [
        "ix_documents_embedding_hnsw",
        "idx_documents_embedding_hnsw",
    ]:
        if index_exists("documents", index_name, inspector):
            op.drop_index(index_name, table_name="documents", schema=schema)

    # Drop embedding column from documents if it exists
    if column_exists("documents", "embedding", inspector):
        op.drop_column("documents", "embedding", schema=schema)


def downgrade() -> None:
    """Restore embedding columns and HNSW indexes, remove chunk_index.

    Note: This downgrade will create empty embedding columns. The actual embeddings
    would need to be restored from a backup or regenerated if rolling back this migration.
    """

    inspector = sa.inspect(op.get_bind())

    # === documents table ===

    # Add embedding column back to documents if it doesn't exist
    if not column_exists("documents", "embedding", inspector):
        op.add_column(
            "documents",
            sa.Column("embedding", Vector(1536), nullable=True),
            schema=schema,
        )

    # Recreate HNSW index on documents.embedding
    if not index_exists("documents", "ix_documents_embedding_hnsw", inspector):
        op.execute(
            f"""
            CREATE INDEX ix_documents_embedding_hnsw
            ON {schema}.documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )

    # === message_embeddings table ===

    # Remove chunk_index column if it exists
    if column_exists("message_embeddings", "chunk_index", inspector):
        op.drop_column("message_embeddings", "chunk_index", schema=schema)

    # Add embedding column back to message_embeddings if it doesn't exist
    if not column_exists("message_embeddings", "embedding", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column("embedding", Vector(1536), nullable=True),
            schema=schema,
        )

    # Recreate HNSW index on message_embeddings.embedding
    if not index_exists(
        "message_embeddings", "ix_message_embeddings_embedding_hnsw", inspector
    ):
        op.execute(
            f"""
            CREATE INDEX ix_message_embeddings_embedding_hnsw
            ON {schema}.message_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
