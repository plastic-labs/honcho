"""merge heads

Revision ID: 66e08642ecd6
Revises: 110bdf470272, baa22cad81e2
Create Date: 2025-11-21 15:40:51.747917

"""

from collections.abc import Sequence

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "66e08642ecd6"
down_revision: str | None = ("110bdf470272", "baa22cad81e2")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
