"""alter vector column dimensions to match VECTOR_STORE.DIMENSIONS

This migration ensures that the embedding columns in message_embeddings and
documents tables use the vector dimension configured in settings rather than
the hardcoded value from previous migrations (1536).

It drops and recreates the HNSW index on documents.embedding since the
index is dimension-specific (HNSW parameters include the vector dimension).

Revision ID: Nj-2KD-Cc2XG
Revises: e4eba9cfaa6f
Create Date: 2026-04-01 19:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "Nj-2KD-Cc2XG"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_atttypmod(table: str, column: str, schema: str | None) -> int | None:
    """Return pg_attribute.atttypmod for a vector column, or None if not found.

    Uses the migration's resolved schema rather than current_schema() to handle
    non-default schema deployments correctly.
    """
    schema_filter = "n.nspname = :schema" if schema else "n.nspname = current_schema()"
    params: dict = {"table": table, "column": column}
    if schema:
        params["schema"] = schema

    result = op.get_bind().execute(
        text(
            f"""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = :table
              AND a.attname = :column
              AND {schema_filter}
            """
        ),
        params,
    )
    row = result.fetchone()
    return int(row[0]) if row else None


def upgrade() -> None:
    """Alter vector columns to match settings.VECTOR_STORE.DIMENSIONS."""
    # Import settings at runtime so we read the deployed configuration
    from src.config import settings

    schema = get_schema()
    target_dim: int = settings.VECTOR_STORE.DIMENSIONS

    inspector = sa.inspect(op.get_bind())

    tables = [
        {
            "name": "documents",
            "column": "embedding",
            "ix_name": "ix_documents_embedding_hnsw",
            "has_ix": None,  # filled below
        },
        {
            "name": "message_embeddings",
            "column": "embedding",
            "ix_name": "ix_message_embeddings_embedding_hnsw",
            "has_ix": None,
        },
    ]

    # Check which indexes actually exist in the DB using inspector dict access
    doc_indexes = {idx["name"] for idx in inspector.get_indexes("documents", schema=schema)}
    msg_indexes = {idx["name"] for idx in inspector.get_indexes("message_embeddings", schema=schema)}
    existing_indexes = doc_indexes | msg_indexes

    for t in tables:
        t["has_ix"] = t["ix_name"] in existing_indexes if t["ix_name"] else False

    for t in tables:
        current_dim = _get_atttypmod(t["name"], t["column"], schema)

        if current_dim is None:
            # Column absent (LanceDB-only backend) — skip
            continue

        if current_dim == target_dim:
            # Already correct
            continue

        # Drop HNSW index before altering column type (HNSW is dimension-typed)
        if t["has_ix"] and t["ix_name"]:
            op.drop_index(t["ix_name"], table_name=t["name"], schema=schema)

        # Build fully-qualified ALTER TABLE statement
        table_ref = f'"{schema}"."{t["name"]}"' if schema else f'"{t["name"]}"'
        op.execute(
            f"ALTER TABLE {table_ref} "
            f'ALTER COLUMN "embedding" TYPE vector({target_dim}) '
            f'USING "embedding"::vector({target_dim})'
        )

        # Recreate HNSW index with new dimension
        if t["has_ix"] and t["ix_name"]:
            op.create_index(
                t["ix_name"],
                t["name"],
                ["embedding"],
                schema=schema,
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "vector_cosine_ops"},
                postgresql_with={"m": 16, "ef_construction": 64},
            )


def downgrade() -> None:
    """Downgrade is not implemented.

    Reducing vector dimensions would truncate existing embedding data.
    A full re-embedding pipeline is required to safely downgrade.
    """
    raise NotImplementedError(
        "Downgrade of vector column dimensions is not supported. "
        "Re-embedding all records with the target dimension is required."
    )
