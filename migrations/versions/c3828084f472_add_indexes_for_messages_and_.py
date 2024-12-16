"""Add indexes for messages and metamessages for reads

Revision ID: c3828084f472
Revises:
Create Date: 2024-12-12 13:41:40.156095

"""

from collections.abc import Sequence
from os import getenv
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3828084f472"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")
    # Add new indexes
    op.create_index(
        "idx_users_app_lookup", "users", ["app_id", "public_id"], schema=schema
    )
    op.create_index(
        "idx_sessions_user_lookup", "sessions", ["user_id", "public_id"], schema=schema
    )

    op.create_index(
        "idx_messages_session_lookup",
        "messages",
        ["session_id", "id"],
        postgresql_include=[
            "public_id",
            "is_user",
            "created_at",
        ],
        schema=schema,
    )

    op.create_index(
        "idx_metamessages_lookup",
        "metamessages",
        ["metamessage_type", sa.text("id DESC")],
        postgresql_include=[
            "public_id",
            "message_id",
            "created_at",
        ],
        schema=schema,
    )


def downgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    # Remove new indexes
    op.drop_index("idx_users_app_lookup", table_name="users", schema=schema)
    op.drop_index("idx_sessions_user_lookup", table_name="sessions", schema=schema)
    op.drop_index("idx_messages_session_lookup", table_name="messages", schema=schema)
    op.drop_index("idx_metamessages_lookup", table_name="metamessages", schema=schema)
