"""make document session_name nullable for sessionless dreams

Allow dreams to run without a session_id by making the session_name
column nullable on the documents table.

Revision ID: e4eba9cfaa6f
Revises: 7c0d9a4e3b1f
Create Date: 2026-01-26

"""

from collections.abc import Sequence

from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "e4eba9cfaa6f"
down_revision: str | None = "119a52b73c60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Make session_name nullable on documents table."""
    op.alter_column("documents", "session_name", nullable=True, schema=schema)


def downgrade() -> None:
    """Make session_name non-nullable again, backfilling NULLs first."""
    op.execute(
        f"UPDATE \"{schema}\".documents SET session_name = '__orphaned__' WHERE session_name IS NULL"
    )
    op.alter_column("documents", "session_name", nullable=False, schema=schema)
