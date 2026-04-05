"""remove Document level_valid constraint

Revision ID: 29ade7350c19
Revises: b8183c5ffb48
Create Date: 2025-11-11 12:54:16.586701

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from migrations.utils import constraint_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "29ade7350c19"
down_revision: str | None = "b8183c5ffb48"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Remove the level_valid CHECK constraint from documents table."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Drop CHECK constraint for level (check both naming convention and plain names)
    # On old databases: "level_valid"
    # On new databases with naming convention: "ck_documents_level_valid"
    if constraint_exists("documents", "level_valid", "check", inspector):
        connection.execute(
            text(f'ALTER TABLE {schema}.documents DROP CONSTRAINT "level_valid"')
        )
    elif constraint_exists("documents", "ck_documents_level_valid", "check", inspector):
        connection.execute(
            text(
                f'ALTER TABLE {schema}.documents DROP CONSTRAINT "ck_documents_level_valid"'
            )
        )


def downgrade() -> None:
    """Restore the level_valid CHECK constraint to documents table."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Normalize any rows with values outside the legacy domain before recreating constraint
    # This ensures downgrade succeeds even if new values like 'implicit' were added after upgrade
    batch_size = 5000
    while True:
        result = connection.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM {schema}.documents
                    WHERE level NOT IN ('explicit', 'deductive')
                    LIMIT :batch_size
                )
                UPDATE {schema}.documents d
                SET level = 'explicit'
                FROM batch
                WHERE d.id = batch.id
                """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Recreate CHECK constraint for level (use plain name to match original migration)
    if not constraint_exists("documents", "level_valid", "check", inspector):
        connection.execute(
            text(
                f"""
                ALTER TABLE {schema}.documents
                ADD CONSTRAINT level_valid
                CHECK (level IN ('explicit', 'deductive'))
                """
            )
        )
