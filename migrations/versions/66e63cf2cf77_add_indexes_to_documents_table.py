"""add hnsw index to documents table

Revision ID: 66e63cf2cf77
Revises: 20f89a421aff
Create Date: 2025-05-19 17:00:18.151735

"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "66e63cf2cf77"
down_revision: str | None = "20f89a421aff"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
schema = settings.DB.SCHEMA


def upgrade() -> None:
    # Create HNSW index on the embedding column for the documents table for cosine distance
    # Parameters:
    # - m: max number of connections (edges) per node (default=16)
    # - ef_construction: size of the candidate list during index construction (default=64)
    print(
        f"Creating HNSW index idx_documents_embedding_hnsw on {schema}.documents table"
    )
    try:
        op.execute(
            text(
                f"""
                CREATE INDEX idx_documents_embedding_hnsw ON {schema}.documents
                USING hnsw (embedding vector_cosine_ops)
                WITH (m=16, ef_construction=64);
                """
            )
        )
        print(
            f"HNSW index idx_documents_embedding_hnsw created on {schema}.documents table"
        )
    except Exception as e:
        print(
            f"Error creating HNSW index idx_documents_embedding_hnsw on {schema}.documents table: {e}"
        )


def downgrade() -> None:
    print(
        f"Dropping HNSW index idx_documents_embedding_hnsw from {schema}.documents table"
    )
    op.execute(text(f"DROP INDEX IF EXISTS {schema}.idx_documents_embedding_hnsw;"))
    print(
        f"HNSW index idx_documents_embedding_hnsw dropped from {schema}.documents table"
    )
