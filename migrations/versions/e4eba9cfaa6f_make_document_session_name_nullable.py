"""make document session_name nullable for sessionless dreams

Allow dreams to run without a session_id by making the session_name
column nullable on the documents table.

Revision ID: e4eba9cfaa6f
Revises: 119a52b73c60
Create Date: 2026-01-26

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from nanoid import generate as generate_nanoid

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "e4eba9cfaa6f"
down_revision: str | None = "119a52b73c60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def ensure_orphaned_sessions_exist(schema: str) -> None:
    """Ensure a per-workspace `__orphaned__` session exists for downgrade.

    During downgrade we backfill NULL `documents.session_name` values to a
    sentinel value (`__orphaned__`) and then make the column non-nullable. The
    `documents` table has a composite foreign key (`session_name`,
    `workspace_name`) that references `sessions` (`name`, `workspace_name`).
    If we set `session_name` to a non-null sentinel without ensuring a matching
    session row exists for each affected workspace, the UPDATE can violate the
    foreign key.

    This helper inserts missing placeholder sessions scoped by workspace using
    `ON CONFLICT DO NOTHING` to remain safe and idempotent.
    """
    conn = op.get_bind()
    orphaned_session_name = "__orphaned__"

    workspaces = conn.execute(
        sa.text(
            f"""
            SELECT DISTINCT workspace_name
            FROM "{schema}".documents
            WHERE session_name IS NULL
            """
        )
    ).fetchall()

    for (workspace_name,) in workspaces:
        conn.execute(
            sa.text(
                f"""
                INSERT INTO "{schema}".sessions (id, name, workspace_name, is_active)
                VALUES (:id, :name, :workspace_name, true)
                ON CONFLICT (name, workspace_name) DO NOTHING
                """
            ),
            {
                "id": generate_nanoid(),
                "name": orphaned_session_name,
                "workspace_name": workspace_name,
            },
        )


def upgrade() -> None:
    """Make session_name nullable on documents table."""
    op.alter_column("documents", "session_name", nullable=True, schema=schema)


def downgrade() -> None:
    """Make session_name non-nullable again, backfilling NULLs first."""
    ensure_orphaned_sessions_exist(schema)
    op.execute(
        f"UPDATE \"{schema}\".documents SET session_name = '__orphaned__' WHERE session_name IS NULL"
    )
    op.alter_column("documents", "session_name", nullable=False, schema=schema)
