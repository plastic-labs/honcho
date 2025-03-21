"""add_keys_table

Revision ID: d8caece57bcf
Revises: c3828084f472
Create Date: 2025-03-21 15:31:31.735401

"""

from collections.abc import Sequence
from os import getenv
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8caece57bcf"
down_revision: Union[str, None] = "c3828084f472"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    op.create_table(
        "keys",
        sa.Column("key", sa.TEXT(), primary_key=True, index=True),
        sa.Column("revoked", sa.Boolean(), nullable=False, default=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema=schema,
    )

    op.create_index("idx_keys_key", "keys", ["key"], unique=True, schema=schema)


def downgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    op.drop_index("idx_keys_key", table_name="keys", schema=schema)
    op.drop_table("keys", schema=schema)
