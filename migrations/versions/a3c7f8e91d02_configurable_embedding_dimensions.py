"""configurable embedding dimensions

Update embedding columns to match the configured VECTOR_STORE.DIMENSIONS.
Previous migrations hardcoded Vector(1536) for OpenAI's default embedding
size. This migration reads the configured dimension at runtime and alters
the columns if they differ, enabling non-OpenAI embedding providers.

Revision ID: a3c7f8e91d02
Revises: e4eba9cfaa6f
Create Date: 2026-04-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "a3c7f8e91d02"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

# Tables and columns that store vector embeddings
EMBEDDING_TABLES = [
    ("message_embeddings", "embedding"),
    ("documents", "embedding"),
]


def _get_current_dimensions(conn: sa.Connection, table: str, column: str) -> int | None:
    """Read the current vector dimension from pg_attribute."""
    result = conn.execute(
        sa.text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = :table_ref::regclass AND attname = :col"
        ),
        {"table_ref": f"{schema}.{table}", "col": column},
    )
    row = result.fetchone()
    return row[0] if row and row[0] > 0 else None


def upgrade() -> None:
    """Alter embedding columns to match configured dimensions."""
    target_dim = settings.VECTOR_STORE.DIMENSIONS
    conn = op.get_bind()

    for table, column in EMBEDDING_TABLES:
        current_dim = _get_current_dimensions(conn, table, column)
        if current_dim == target_dim:
            print(
                f"Column {schema}.{table}.{column} already has "
                f"{target_dim} dimensions, skipping"
            )
            continue

        print(
            f"Altering {schema}.{table}.{column} from "
            f"{current_dim} to {target_dim} dimensions"
        )

        # Drop HNSW index if it exists (cannot alter column type with index)
        index_name = f"idx_{table}_embedding_hnsw"
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {schema}.{index_name}"))

        # Alter column type
        conn.execute(
            sa.text(
                f"ALTER TABLE {schema}.{table} "
                f"ALTER COLUMN {column} TYPE vector({target_dim})"
            )
        )

        # Recreate HNSW index
        if table == "documents":
            conn.execute(
                sa.text(
                    f"CREATE INDEX {index_name} ON {schema}.{table} "
                    f"USING hnsw ({column} vector_cosine_ops)"
                )
            )


def downgrade() -> None:
    """Revert embedding columns to 1536 dimensions (OpenAI default)."""
    conn = op.get_bind()

    for table, column in EMBEDDING_TABLES:
        current_dim = _get_current_dimensions(conn, table, column)
        if current_dim == 1536:
            continue

        index_name = f"idx_{table}_embedding_hnsw"
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {schema}.{index_name}"))

        conn.execute(
            sa.text(
                f"ALTER TABLE {schema}.{table} ALTER COLUMN {column} TYPE vector(1536)"
            )
        )

        if table == "documents":
            conn.execute(
                sa.text(
                    f"CREATE INDEX {index_name} ON {schema}.{table} "
                    f"USING hnsw ({column} vector_cosine_ops)"
                )
            )
