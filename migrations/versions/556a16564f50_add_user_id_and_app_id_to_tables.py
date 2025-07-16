"""add user id and app id to tables

Revision ID: 556a16564f50
Revises: b765d82110bd
Create Date: 2025-05-13 17:10:33.805495

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.exc import IntegrityError, ProgrammingError

from migrations.utils import (
    column_exists,
    fk_exists,
    get_schema,
    index_exists,
)

# revision identifiers, used by Alembic.
revision: str = "556a16564f50"
down_revision: str | None = "b765d82110bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    schema = get_schema()
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # 1. SESSIONS TABLE
    if not column_exists("sessions", "app_id", inspector):
        op.add_column(
            "sessions", sa.Column("app_id", sa.Text(), nullable=True), schema=schema
        )
        print("Added app_id column to sessions table")
    else:
        print("app_id column already exists in sessions table")

    # Create foreign key constraint for app_id
    if not fk_exists("sessions", "sessions_app_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "sessions_app_id_fkey",
                "sessions",
                "apps",
                ["app_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created app_id foreign key for sessions table")
        except IntegrityError:
            print(
                "Cannot create app_id foreign key for sessions table - integrity error"
            )
    else:
        print("app_id foreign key already exists for sessions table")

    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE sessions s
            SET app_id = u.app_id
            FROM users u
            WHERE s.user_id = u.public_id
            AND (s.app_id IS NULL OR s.app_id <> u.app_id)
        """)
        print("Updated app_id values from users for sessions table")
    except Exception as e:
        print(f"Error updating app_id values for sessions table: {e}")

    # Make app_id non-nullable
    try:
        op.alter_column("sessions", "app_id", nullable=False, schema=schema)
        print("Made app_id non-nullable for sessions table")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making app_id non-nullable for sessions table: {e}")

    # Create index for sessions.app_id
    if not index_exists("sessions", "ix_sessions_app_id"):
        op.create_index(
            op.f("ix_sessions_app_id"),
            "sessions",
            ["app_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_sessions_app_id")
    else:
        print("Index ix_sessions_app_id already exists")

    # 2. MESSAGES TABLE
    if not column_exists("messages", "app_id", inspector):
        op.add_column("messages", sa.Column("app_id", sa.Text()), schema=schema)
        print("Added app_id column to messages table")
    else:
        print("app_id column already exists in messages table")

    if not column_exists("messages", "user_id", inspector):
        op.add_column("messages", sa.Column("user_id", sa.Text()), schema=schema)
        print("Added user_id column to messages table")
    else:
        print("user_id column already exists in messages table")

    # Create foreign key constraints for app_id and user_id
    if not fk_exists("messages", "messages_app_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "messages_app_id_fkey",
                "messages",
                "apps",
                ["app_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created app_id foreign key for messages table")
        except IntegrityError:
            print(
                "Cannot create app_id foreign key for messages table - integrity error"
            )
    else:
        print("app_id foreign key already exists for messages table")

    if not fk_exists("messages", "messages_user_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "messages_user_id_fkey",
                "messages",
                "users",
                ["user_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created user_id foreign key for messages table")
        except IntegrityError:
            print(
                "Cannot create user_id foreign key for messages table - integrity error"
            )
    else:
        print("user_id foreign key already exists for messages table")

    # Data migration: Fill app_id and user_id from sessions table
    try:
        op.execute("""
            UPDATE messages m
            SET app_id = s.app_id, user_id = s.user_id
            FROM sessions s
            WHERE m.session_id = s.public_id
            AND (
                m.app_id IS NULL OR m.app_id <> s.app_id OR
                m.user_id IS NULL OR m.user_id <> s.user_id
            )
        """)
        print("Updated app_id and user_id values from sessions for messages table")
    except Exception as e:
        print(f"Error updating app_id and user_id values for messages table: {e}")

    # Make app_id and user_id non-nullable
    try:
        op.alter_column("messages", "app_id", nullable=False, schema=schema)
        op.alter_column("messages", "user_id", nullable=False, schema=schema)
        print("Made app_id and user_id non-nullable for messages table")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making app_id and user_id non-nullable for messages table: {e}")

    # Create indices for messages foreign keys
    if not index_exists("messages", "ix_messages_app_id"):
        op.create_index(
            op.f("ix_messages_app_id"),
            "messages",
            ["app_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_messages_app_id")
    else:
        print("Index ix_messages_app_id already exists")

    if not index_exists("messages", "ix_messages_user_id"):
        op.create_index(
            op.f("ix_messages_user_id"),
            "messages",
            ["user_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_messages_user_id")
    else:
        print("Index ix_messages_user_id already exists")

    # 3. METAMESSAGES TABLE
    if not column_exists("metamessages", "app_id", inspector):
        op.add_column("metamessages", sa.Column("app_id", sa.Text()), schema=schema)
        print("Added app_id column to metamessages table")
    else:
        print("app_id column already exists in metamessages table")

    # Create foreign key constraint for app_id
    if not fk_exists("metamessages", "metamessages_app_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "metamessages_app_id_fkey",
                "metamessages",
                "apps",
                ["app_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created app_id foreign key for metamessages table")
        except IntegrityError:
            print(
                "Cannot create app_id foreign key for metamessages table - integrity error"
            )
    else:
        print("app_id foreign key already exists for metamessages table")

    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE metamessages m
            SET app_id = u.app_id
            FROM users u
            WHERE m.user_id = u.public_id
            AND (m.app_id IS NULL OR m.app_id <> u.app_id)
        """)
        print("Updated app_id values from users")
    except Exception as e:
        print(f"Error updating app_id values: {e}")

    # Make app_id non-nullable
    try:
        op.alter_column("metamessages", "app_id", nullable=False, schema=schema)
        print("Made app_id non-nullable for metamessages table")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making app_id non-nullable for metamessages table: {e}")

    # Create index for metamessages.app_id
    if not index_exists("metamessages", "ix_metamessages_app_id"):
        op.create_index(
            op.f("ix_metamessages_app_id"),
            "metamessages",
            ["app_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_metamessages_app_id")
    else:
        print("Index ix_metamessages_app_id already exists")

    # 4. COLLECTIONS TABLE
    if not column_exists("collections", "app_id", inspector):
        op.add_column("collections", sa.Column("app_id", sa.Text()), schema=schema)
        print("Added app_id column to collections table")
    else:
        print("app_id column already exists in collections table")

    # Create foreign key constraint for app_id
    if not fk_exists("collections", "collections_app_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "collections_app_id_fkey",
                "collections",
                "apps",
                ["app_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created app_id foreign key for collections table")
        except IntegrityError:
            print(
                "Cannot create app_id foreign key for collections table - integrity error"
            )
    else:
        print("app_id foreign key already exists for collections table")

    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE collections c
            SET app_id = u.app_id
            FROM users u
            WHERE c.user_id = u.public_id
            AND (c.app_id IS NULL OR c.app_id <> u.app_id)
        """)
        print("Updated app_id values from users for collections table")
    except Exception as e:
        print(f"Error updating app_id values for collections table: {e}")

    # Make app_id non-nullable
    try:
        op.alter_column("collections", "app_id", nullable=False, schema=schema)
        print("Made app_id non-nullable for collections table")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making app_id non-nullable for collections table: {e}")

    # Create index for collections.app_id
    if not index_exists("collections", "ix_collections_app_id"):
        op.create_index(
            op.f("ix_collections_app_id"),
            "collections",
            ["app_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_collections_app_id")
    else:
        print("Index ix_collections_app_id already exists")

    # 5. DOCUMENTS TABLE
    if not column_exists("documents", "app_id", inspector):
        op.add_column("documents", sa.Column("app_id", sa.Text()), schema=schema)
        print("Added app_id column to documents table")
    else:
        print("app_id column already exists in documents table")

    if not column_exists("documents", "user_id", inspector):
        op.add_column("documents", sa.Column("user_id", sa.Text()), schema=schema)
        print("Added user_id column to documents table")
    else:
        print("user_id column already exists in documents table")

    # Create foreign key constraints for app_id and user_id
    if not fk_exists("documents", "documents_app_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "documents_app_id_fkey",
                "documents",
                "apps",
                ["app_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created app_id foreign key for documents table")
        except IntegrityError:
            print(
                "Cannot create app_id foreign key for documents table - integrity error"
            )
    else:
        print("app_id foreign key already exists for documents table")

    if not fk_exists("documents", "documents_user_id_fkey", inspector):
        try:
            op.create_foreign_key(
                "documents_user_id_fkey",
                "documents",
                "users",
                ["user_id"],
                ["public_id"],
                source_schema=schema,
                referent_schema=schema,
            )
            print("Created user_id foreign key for documents table")
        except IntegrityError:
            print(
                "Cannot create user_id foreign key for documents table - integrity error"
            )
    else:
        print("user_id foreign key already exists for documents table")

    # Data migration: Fill app_id and user_id from collections table
    try:
        op.execute("""
            UPDATE documents d
            SET app_id = c.app_id, user_id = c.user_id
            FROM collections c
            WHERE d.collection_id = c.public_id
            AND (
                d.app_id IS NULL OR d.app_id <> c.app_id OR
                d.user_id IS NULL OR d.user_id <> c.user_id
            )
        """)
        print("Updated app_id and user_id values from collections for documents table")
    except Exception as e:
        print(f"Error updating app_id and user_id values for documents table: {e}")

    # Make app_id and user_id non-nullable
    try:
        op.alter_column("documents", "app_id", nullable=False, schema=schema)
        op.alter_column("documents", "user_id", nullable=False, schema=schema)
        print("Made app_id and user_id non-nullable for documents table")
    except (ProgrammingError, IntegrityError) as e:
        print(f"Error making app_id and user_id non-nullable for documents table: {e}")

    # Create indices for documents foreign keys
    if not index_exists("documents", "ix_documents_app_id"):
        op.create_index(
            op.f("ix_documents_app_id"),
            "documents",
            ["app_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_documents_app_id")
    else:
        print("Index ix_documents_app_id already exists")

    if not index_exists("documents", "ix_documents_user_id"):
        op.create_index(
            op.f("ix_documents_user_id"),
            "documents",
            ["user_id"],
            unique=False,
            schema=schema,
        )
        print("Created index ix_documents_user_id")
    else:
        print("Index ix_documents_user_id already exists")


def downgrade():
    schema = get_schema()
    inspector = sa.inspect(op.get_bind())

    # 5. Documents table

    # Drop indices
    if index_exists("documents", "ix_documents_app_id", inspector):
        op.drop_index("ix_documents_app_id", table_name="documents", schema=schema)
        print("Dropped index ix_documents_app_id")
    if index_exists("documents", "ix_documents_user_id", inspector):
        op.drop_index("ix_documents_user_id", table_name="documents", schema=schema)
        print("Dropped index ix_documents_user_id")

    # Make app_id and user_id nullable again
    op.alter_column("documents", "app_id", nullable=True, schema=schema)
    op.alter_column("documents", "user_id", nullable=True, schema=schema)
    print("Made app_id and user_id nullable again for documents table")

    # Drop foreign key constraints
    try:
        op.drop_constraint("documents_user_id_fkey", "documents", schema=schema)
        print("Dropped user_id foreign key for documents table")
    except Exception as e:
        print(f"Error dropping user_id foreign key for documents table: {e}")

    try:
        op.drop_constraint("documents_app_id_fkey", "documents", schema=schema)
        print("Dropped app_id foreign key for documents table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for documents table: {e}")

    # Drop the columns
    op.drop_column("documents", "user_id", schema=schema)
    op.drop_column("documents", "app_id", schema=schema)
    print("Dropped app_id and user_id columns from documents table")

    # 4. Collections table

    # Drop indices
    if index_exists("collections", "ix_collections_app_id", inspector):
        op.drop_index("ix_collections_app_id", table_name="collections", schema=schema)
        print("Dropped index ix_collections_app_id")

    # Make app_id nullable again
    op.alter_column("collections", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for collections table")

    # Drop foreign key constraint
    try:
        op.drop_constraint("collections_app_id_fkey", "collections", schema=schema)
        print("Dropped app_id foreign key for collections table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for collections table: {e}")

    # Drop the column
    op.drop_column("collections", "app_id", schema=schema)
    print("Dropped app_id column from collections table")

    # 3. Metamessages table

    # Drop indices
    if index_exists("metamessages", "ix_metamessages_app_id", inspector):
        op.drop_index(
            "ix_metamessages_app_id", table_name="metamessages", schema=schema
        )
        print("Dropped index ix_metamessages_app_id")

    # Make app_id nullable again
    op.alter_column("metamessages", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for metamessages table")

    # Drop foreign key constraint
    try:
        op.drop_constraint("metamessages_app_id_fkey", "metamessages", schema=schema)
        print("Dropped app_id foreign key for metamessages table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for metamessages table: {e}")

    # Drop the column
    op.drop_column("metamessages", "app_id", schema=schema)
    print("Dropped app_id column from metamessages table")

    # 2. Messages table

    # Drop indices
    if index_exists("messages", "ix_messages_app_id", inspector):
        op.drop_index("ix_messages_app_id", table_name="messages", schema=schema)
        print("Dropped index ix_messages_app_id")
    if index_exists("messages", "ix_messages_user_id", inspector):
        op.drop_index("ix_messages_user_id", table_name="messages", schema=schema)
        print("Dropped index ix_messages_user_id")

    # Make app_id and user_id nullable again
    op.alter_column("messages", "app_id", nullable=True, schema=schema)
    op.alter_column("messages", "user_id", nullable=True, schema=schema)
    print("Made app_id and user_id nullable again for messages table")

    # Drop foreign key constraints
    try:
        op.drop_constraint("messages_user_id_fkey", "messages", schema=schema)
        print("Dropped user_id foreign key for messages table")
    except Exception as e:
        print(f"Error dropping user_id foreign key for messages table: {e}")

    try:
        op.drop_constraint("messages_app_id_fkey", "messages", schema=schema)
        print("Dropped app_id foreign key for messages table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for messages table: {e}")

    # Drop the columns
    op.drop_column("messages", "user_id", schema=schema)
    op.drop_column("messages", "app_id", schema=schema)
    print("Dropped app_id and user_id columns from messages table")

    # 1. Sessions table

    # Drop indices
    if index_exists("sessions", "ix_sessions_app_id", inspector):
        op.drop_index("ix_sessions_app_id", table_name="sessions", schema=schema)
        print("Dropped index ix_sessions_app_id")

    # Make app_id nullable again
    op.alter_column("sessions", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for sessions table")

    # Drop foreign key constraint
    try:
        op.drop_constraint("sessions_app_id_fkey", "sessions", schema=schema)
        print("Dropped app_id foreign key for sessions table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for sessions table: {e}")

    # Drop the column
    op.drop_column("sessions", "app_id", schema=schema)
    print("Dropped app_id column from sessions table")
