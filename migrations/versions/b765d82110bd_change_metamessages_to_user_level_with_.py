"""Change Metamessages to user level with optional message and session link

Revision ID: b765d82110bd
Revises: c3828084f472
Create Date: 2025-04-03 15:32:16.733312

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b765d82110bd"
down_revision: Union[str, None] = "c3828084f472"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add new columns to metamessages table
    op.add_column("metamessages", sa.Column("user_id", sa.TEXT(), nullable=True))
    op.add_column("metamessages", sa.Column("session_id", sa.TEXT(), nullable=True))

    # 2. Create foreign key constraints for the new columns
    op.create_foreign_key(
        "fk_metamessages_user_id_users",
        "metamessages",
        "users",
        ["user_id"],
        ["public_id"],
    )
    op.create_foreign_key(
        "fk_metamessages_session_id_sessions",
        "metamessages",
        "sessions",
        ["session_id"],
        ["public_id"],
    )

    # 3. Make message_id nullable
    op.alter_column(
        "metamessages", "message_id", existing_type=sa.TEXT(), nullable=True
    )

    # 4. Create indices for users, sessions, and messages - only if they don't exist

    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Check and create each index only if it doesn't exist
    existing_indices = {
        idx["name"]
        for schema in [None, "public"]
        for tbl in inspector.get_table_names(schema=schema)
        for idx in inspector.get_indexes(tbl, schema=schema)
    }

    # Helper function to create index if not exists
    def create_index_if_not_exists(index_name, table_name, columns, **kwargs):
        if index_name not in existing_indices:
            op.create_index(index_name, table_name, columns, **kwargs)

    # Create all indices with the helper function
    create_index_if_not_exists("idx_users_app_lookup", "users", ["app_id", "public_id"])
    create_index_if_not_exists(
        "idx_sessions_user_lookup", "sessions", ["user_id", "public_id"]
    )

    # For complex indices
    create_index_if_not_exists(
        "idx_messages_session_lookup",
        "messages",
        ["session_id", "id"],
        postgresql_include=["public_id", "is_user", "created_at"],
    )

    create_index_if_not_exists(
        "idx_metamessages_lookup",
        "metamessages",
        ["metamessage_type", text("id DESC")],
        postgresql_include=["public_id", "message_id", "created_at"],
    )

    create_index_if_not_exists(
        "idx_metamessages_user_lookup",
        "metamessages",
        ["user_id", "metamessage_type", text("id DESC")],
    )

    create_index_if_not_exists(
        "idx_metamessages_session_lookup",
        "metamessages",
        ["session_id", "metamessage_type", text("id DESC")],
    )

    create_index_if_not_exists(
        "idx_metamessages_message_lookup",
        "metamessages",
        ["message_id", "metamessage_type", text("id DESC")],
    )

    # 7. Add the check constraint for message_id and session_id relationship
    op.create_check_constraint(
        "message_requires_session",
        "metamessages",
        "(message_id IS NULL) OR (session_id IS NOT NULL)",
    )

    # 8. Update existing data: fill user_id from message's session's user
    # This is a complex data migration that requires SQL
    op.execute("""
        UPDATE metamessages m
        SET user_id = u.public_id, 
            session_id = s.public_id
        FROM messages msg
        JOIN sessions s ON msg.session_id = s.public_id
        JOIN users u ON s.user_id = u.public_id
        WHERE m.message_id = msg.public_id
        AND m.user_id IS NULL
    """)

    # 9. Now that data is migrated, make user_id not nullable
    op.alter_column("metamessages", "user_id", existing_type=sa.TEXT(), nullable=False)


def downgrade() -> None:
    # 1. Remove the check constraint
    op.drop_constraint("message_requires_session", "metamessages", type_="check")

    # 2. Drop all the new indices
    op.drop_index("idx_metamessages_message_lookup", table_name="metamessages")
    op.drop_index("idx_metamessages_session_lookup", table_name="metamessages")
    op.drop_index("idx_metamessages_user_lookup", table_name="metamessages")
    op.drop_index("idx_metamessages_lookup", table_name="metamessages")
    op.drop_index("idx_messages_session_lookup", table_name="messages")
    op.drop_index("idx_sessions_user_lookup", table_name="sessions")
    op.drop_index("idx_users_app_lookup", table_name="users")

    # 3. Remove foreign key constraints
    op.drop_constraint(
        "fk_metamessages_session_id_sessions", "metamessages", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_metamessages_user_id_users", "metamessages", type_="foreignkey"
    )

    # 4. Make message_id required again and clean up data if needed
    op.execute("""
        DELETE FROM metamessages WHERE message_id IS NULL
    """)
    op.alter_column(
        "metamessages", "message_id", existing_type=sa.TEXT(), nullable=False
    )

    # 5. Drop the new columns
    op.drop_column("metamessages", "session_id")
    op.drop_column("metamessages", "user_id")
