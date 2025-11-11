"""remove Document level_valid constraint

Revision ID: 29ade7350c19
Revises: b8183c5ffb48
Create Date: 2025-11-11 12:54:16.586701

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

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

    # Drop CHECK constraint for level
    if constraint_exists("documents", "level_valid", "check", inspector):
        op.drop_constraint(
            "level_valid",
            "documents",
            type_="check",
            schema=schema,
        )


def downgrade() -> None:
    """Restore the level_valid CHECK constraint to documents table."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Recreate CHECK constraint for level
    if not constraint_exists("documents", "level_valid", "check", inspector):
        op.create_check_constraint(
            "level_valid",
            "documents",
            "level IN ('explicit', 'deductive')",
            schema=schema,
        )
