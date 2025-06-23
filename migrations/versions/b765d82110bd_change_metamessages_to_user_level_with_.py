"""Change Metamessages to user level with optional message and session link

Revision ID: b765d82110bd
Revises: c3828084f472
Create Date: 2025-04-03 15:32:16.733312

"""

from collections.abc import Sequence
from typing import Union
from os import getenv

import sqlalchemy as sa
from alembic import op
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.sql import text

from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "b765d82110bd"
down_revision: str | None = "c3828084f472"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    schema = settings.DB.SCHEMA

    conn = op.get_bind()

    # Check which columns already exist in the metamessages table
    inspector = sa.inspect(conn)
    existing_columns = [col["name"] for col in inspector.get_columns("metamessages")]

    # 1. Add new columns to metamessages table if they don't exist
    if "user_id" not in existing_columns:
        op.add_column(
            "metamessages",
            sa.Column("user_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        print("Added user_id column")
    else:
        print("user_id column already exists")

    if "session_id" not in existing_columns:
        op.add_column(
            "metamessages",
            sa.Column("session_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        print("Added session_id column")
    else:
        print("session_id column already exists")

    # 2. Create foreign key constraints for the new columns
    # Check if foreign key constraints already exist
    foreign_keys = inspector.get_foreign_keys("metamessages")
    fk_names = [fk.get("name") for fk in foreign_keys]

    if "fk_metamessages_user_id_users" not in fk_names:
        try:
            op.create_foreign_key(
                "fk_metamessages_user_id_users",
                "metamessages",
                "users",
                ["user_id"],
                ["public_id"],
            )
            print("Created user_id foreign key")
        except IntegrityError:
            print("Cannot create user_id foreign key - integrity error")

    if "fk_metamessages_session_id_sessions" not in fk_names:
        try:
            op.create_foreign_key(
                "fk_metamessages_session_id_sessions",
                "metamessages",
                "sessions",
                ["session_id"],
                ["public_id"],
            )
            print("Created session_id foreign key")
        except IntegrityError:
            print("Cannot create session_id foreign key - integrity error")

    # 3. Make message_id nullable if it's not already
    try:
        op.alter_column(
            "metamessages",
            "message_id",
            existing_type=sa.TEXT(),
            nullable=True,
            schema=schema,
        )
        print("Made message_id nullable")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making message_id nullable: {e}")

    # 4. Data migration: Fill session_id from messages
    try:
        op.execute(
            """
            UPDATE metamessages m
            SET session_id = msg.session_id
            FROM messages msg
            WHERE m.message_id = msg.public_id
            AND m.session_id IS NULL
        """
        )
        print("Updated session_id values from messages")
    except Exception as e:
        print(f"Error updating session_id values: {e}")

    # 5. Handle orphaned records - records with message_id but no session_id
    try:
        orphaned_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM metamessages WHERE message_id IS NOT NULL AND session_id IS NULL"
            )
        ).scalar()

        if orphaned_count and orphaned_count > 0:
            print(
                f"Found {orphaned_count} orphaned records. Setting message_id to NULL."
            )
            op.execute(
                """
                UPDATE metamessages
                SET message_id = NULL
                WHERE message_id IS NOT NULL AND session_id IS NULL
            """
            )
    except Exception as e:
        print(f"Error handling orphaned records: {e}")

    # 6. Data migration: Fill user_id from session's user
    try:
        op.execute(
            """
            UPDATE metamessages m
            SET user_id = s.user_id
            FROM sessions s
            WHERE m.session_id = s.public_id
            AND m.user_id IS NULL
        """
        )
        print("Updated user_id values from sessions")
    except Exception as e:
        print(f"Error updating user_id values from sessions: {e}")

    # 7. Create indices for the various lookups - only if they don't exist
    existing_indices = {
        idx["name"]
        for tbl in inspector.get_table_names(schema=schema)
        if tbl in ["metamessages", "users", "sessions", "messages"]
        for idx in inspector.get_indexes(tbl, schema=schema)
    }

    # Helper function to create index if not exists
    def create_index_if_not_exists(index_name, table_name, columns, **kwargs):
        if index_name not in existing_indices:
            try:
                op.create_index(index_name, table_name, columns, **kwargs)
                print(f"Created index {index_name}")
            except Exception as e:
                print(f"Error creating index {index_name}: {e}")

    create_index_if_not_exists(
        "idx_metamessages_user_lookup",
        "metamessages",
        ["user_id", "metamessage_type", text("id DESC")],
        schema=schema,
    )

    create_index_if_not_exists(
        "idx_metamessages_session_lookup",
        "metamessages",
        ["session_id", "metamessage_type", text("id DESC")],
        schema=schema,
    )

    create_index_if_not_exists(
        "idx_metamessages_message_lookup",
        "metamessages",
        ["message_id", "metamessage_type", text("id DESC")],
        schema=schema,
    )

    # 8. Verify the data is ready for the constraint
    constraint_violations = conn.execute(
        text(
            "SELECT COUNT(*) FROM metamessages WHERE message_id IS NOT NULL AND session_id IS NULL"
        )
    ).scalar()

    if constraint_violations and constraint_violations > 0:
        print(f"WARNING: {constraint_violations} records would violate the constraint!")
        print("Setting these message_ids to NULL to prevent constraint violation")
        op.execute(
            """
            UPDATE metamessages
            SET message_id = NULL
            WHERE message_id IS NOT NULL AND session_id IS NULL
        """
        )

    # 9. Add the check constraint for message_id and session_id relationship
    try:
        # Check if constraint already exists
        constraint_exists = (
            conn.execute(
                text(
                    "SELECT 1 FROM pg_constraint WHERE conname = 'message_requires_session'"
                )
            ).fetchone()
            is not None
        )

        if not constraint_exists:
            op.create_check_constraint(
                "message_requires_session",
                "metamessages",
                "(message_id IS NULL) OR (session_id IS NOT NULL)",
            )
            print("Created message_requires_session constraint")
        else:
            print("message_requires_session constraint already exists")
    except Exception as e:
        print(f"Error creating constraint: {e}")

    # 10. Now that data is migrated, make user_id not nullable if it has values
    # First ensure all records have a user_id
    null_user_count = conn.execute(
        text("SELECT COUNT(*) FROM metamessages WHERE user_id IS NULL")
    ).scalar()

    if null_user_count and null_user_count > 0:
        print(f"WARNING: {null_user_count} records have NULL user_id!")
        # Try to derive user_id from message for these orphaned records
        op.execute(
            """
            UPDATE metamessages m
            SET user_id = u.public_id
            FROM messages msg
            JOIN sessions s ON msg.session_id = s.public_id
            JOIN users u ON s.user_id = u.public_id
            WHERE m.message_id = msg.public_id
            AND m.user_id IS NULL
        """
        )

        # Check again after fixes
        null_user_count = conn.execute(
            text("SELECT COUNT(*) FROM metamessages WHERE user_id IS NULL")
        ).scalar()

        if null_user_count == 0:
            try:
                op.alter_column(
                    "metamessages",
                    "user_id",
                    existing_type=sa.TEXT(),
                    nullable=False,
                    schema=schema,
                )
                print("Made user_id not nullable")
            except Exception as e:
                print(f"Error making user_id not nullable: {e}")
        else:
            print(
                f"Still have {null_user_count} records with NULL user_id, cannot make column not nullable"
            )
    else:
        try:
            op.alter_column(
                "metamessages",
                "user_id",
                existing_type=sa.TEXT(),
                nullable=False,
                schema=schema,
            )
            print("Made user_id not nullable")
        except Exception as e:
            print(f"Error making user_id not nullable: {e}")


def downgrade() -> None:
    schema = settings.DB.SCHEMA
    # Add try-except blocks to handle case where elements don't exist

    # 1. Remove the check constraint
    try:
        op.drop_constraint(
            "message_requires_session", "metamessages", type_="check", schema=schema
        )
    except Exception as e:
        print(f"Error dropping message_requires_session constraint: {e}")

    # 2. Drop all the new indices
    for index_name, table_name in [
        ("idx_metamessages_message_lookup", "metamessages"),
        ("idx_metamessages_session_lookup", "metamessages"),
        ("idx_metamessages_user_lookup", "metamessages"),
    ]:
        try:
            op.drop_index(index_name, table_name=table_name, schema=schema)
        except Exception as e:
            print(f"Error dropping index {index_name}: {e}")

    # 3. Remove foreign key constraints
    try:
        op.drop_constraint(
            "fk_metamessages_session_id_sessions",
            "metamessages",
            type_="foreignkey",
            schema=schema,
        )
    except Exception as e:
        print(f"Error dropping session_id foreign key: {e}")

    try:
        op.drop_constraint(
            "fk_metamessages_user_id_users",
            "metamessages",
            type_="foreignkey",
            schema=schema,
        )
    except Exception as e:
        print(f"Error dropping user_id foreign key: {e}")

    # 4. Make message_id required again and clean up data if needed
    try:
        op.execute(
            """
            DELETE FROM metamessages WHERE message_id IS NULL
        """
        )
        op.alter_column(
            "metamessages",
            "message_id",
            existing_type=sa.TEXT(),
            nullable=False,
            schema=schema,
        )
    except Exception as e:
        print(f"Error making message_id not nullable: {e}")

    # 5. Drop the new columns
    try:
        op.drop_column("metamessages", "session_id", schema=schema)
    except Exception as e:
        print(f"Error dropping session_id column: {e}")

    try:
        op.drop_column("metamessages", "user_id", schema=schema)
    except Exception as e:
        print(f"Error dropping user_id column: {e}")
