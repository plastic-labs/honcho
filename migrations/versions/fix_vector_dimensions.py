"""Update vector columns to use configurable dimensions.

Revision ID: a1b2c3d4e5f7
Revises: e4eba9cfaa6f
Create Date: 2026-04-01
"""

from alembic import op
from sqlalchemy import text

revision = "a1b2c3d4e5f7"
down_revision = "e4eba9cfaa6f"
branch_labels = None
depends_on = None


def get_configured_dimensions():
    import os
    return int(os.environ.get("VECTOR_STORE_DIMENSIONS", "1536"))


def _get_current_vector_dims(table: str, column: str) -> int | None:
    """Query pg_attribute for the current vector column dimensions."""
    conn = op.get_bind()
    row = conn.execute(
        text(
            "SELECT atttypmod FROM pg_attribute "
            "JOIN pg_class ON attrelid = pg_class.oid "
            "JOIN pg_namespace ON relnamespace = pg_namespace.oid "
            "WHERE relname = :table AND attname = :column "
            "AND nspname = current_schema()"
        ),
        {"table": table, "column": column},
    ).fetchone()
    if row is None:
        return None
    return row[0]


def upgrade() -> None:
    dims = get_configured_dimensions()
    for table, column in [
        ("message_embeddings", "embedding"),
        ("documents", "embedding"),
    ]:
        current = _get_current_vector_dims(table, column)
        if current == dims:
            continue
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({dims})")


def downgrade() -> None:
    for table, column in [
        ("message_embeddings", "embedding"),
        ("documents", "embedding"),
    ]:
        current = _get_current_vector_dims(table, column)
        if current == 1536:
            continue
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector(1536)")
