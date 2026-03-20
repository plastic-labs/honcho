"""configurable_embedding_dimensions

Make embedding vector columns match VECTOR_STORE_DIMENSIONS config.
Nulls out existing embeddings and marks them pending so the reconciler
re-embeds them with the active provider and dimensions.

Only runs if the current column dimensions differ from config —
existing 1536-dim deployments that haven't changed config are untouched.

Revision ID: c92bda362202
Revises: f1a2b3c4d5e6
Create Date: 2026-03-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, table_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "c92bda362202"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

# Tables with embedding columns: (table_name, column_name)
_EMBEDDING_TABLES = [
    ("message_embeddings", "embedding"),
    ("documents", "embedding"),
]


def _get_current_vector_dim(
    connection: sa.Connection, table: str, column: str
) -> int | None:
    """Query the pg column type to extract the current vector dimension."""
    result = connection.execute(
        sa.text(
            "SELECT atttypmod FROM pg_attribute "
            "JOIN pg_class ON pg_attribute.attrelid = pg_class.oid "
            "JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid "
            "WHERE pg_namespace.nspname = :schema "
            "AND pg_class.relname = :table "
            "AND pg_attribute.attname = :column"
        ),
        {"schema": schema or "public", "table": table, "column": column},
    )
    row = result.fetchone()
    if row and row[0] and row[0] > 0:
        return row[0]
    return None


def upgrade() -> None:
    """Resize embedding columns if VECTOR_STORE_DIMENSIONS differs from DB."""
    target_dims = settings.VECTOR_STORE.DIMENSIONS
    connection = op.get_bind()

    for table, column in _EMBEDDING_TABLES:
        if not table_exists(table):
            continue

        current_dims = _get_current_vector_dim(connection, table, column)
        if current_dims == target_dims:
            continue

        old_label = f"{current_dims}d" if current_dims else "unknown"
        op.execute(
            sa.text(
                f"UPDATE {schema + '.' if schema else ''}{table} "
                f"SET {column} = NULL, sync_state = 'pending' "
                f"WHERE {column} IS NOT NULL"
            )
        )

        op.execute(
            sa.text(
                f"ALTER TABLE {schema + '.' if schema else ''}{table} "
                f"ALTER COLUMN {column} TYPE vector({target_dims})"
            )
        )

        print(
            f"  ✓ {table}.{column}: {old_label} → {target_dims}d "
            f"(embeddings nulled, reconciler will re-embed)"
        )


def downgrade() -> None:
    """Revert to 1536 dimensions (the original default)."""
    connection = op.get_bind()

    for table, column in _EMBEDDING_TABLES:
        if not table_exists(table):
            continue

        current_dims = _get_current_vector_dim(connection, table, column)
        if current_dims == 1536:
            continue

        op.execute(
            sa.text(
                f"UPDATE {schema + '.' if schema else ''}{table} "
                f"SET {column} = NULL, sync_state = 'pending' "
                f"WHERE {column} IS NOT NULL"
            )
        )

        op.execute(
            sa.text(
                f"ALTER TABLE {schema + '.' if schema else ''}{table} "
                f"ALTER COLUMN {column} TYPE vector(1536)"
            )
        )
