"""merge migration heads

Revision ID: 121203eca967
Revises: 110bdf470272, b8183c5ffb48
Create Date: 2025-11-12 18:04:42.495898

"""

from collections.abc import Sequence

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "121203eca967"
down_revision: str | None = ("110bdf470272", "b8183c5ffb48")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
