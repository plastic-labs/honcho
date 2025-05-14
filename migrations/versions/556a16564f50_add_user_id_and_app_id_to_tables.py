"""add user id and app id to tables

Revision ID: 556a16564f50
Revises: b765d82110bd
Create Date: 2025-05-13 17:10:33.805495

"""
from typing import Sequence, Union
from os import getenv

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, ProgrammingError

# revision identifiers, used by Alembic.
revision: str = '556a16564f50'
down_revision: Union[str, None] = 'b765d82110bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    # 1. Sessions table
    # Add app_id to sessions table
    op.add_column("sessions", sa.Column("app_id", sa.Text(), nullable=True), schema=schema)
    print("Added app_id column to sessions table")

    # Create foreign key constraint for app_id
    try:
        op.create_foreign_key(
            "fk_sessions_app_id_apps",
            "sessions",
            "apps",
            ["app_id"],
            ["public_id"]
        )
        print("Created app_id foreign key")
    except IntegrityError:
        print("Cannot create app_id foreign key - integrity error")

    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE sessions s
            SET app_id = u.app_id
            FROM users u
            WHERE s.user_id = u.public_id
            AND s.app_id IS NULL
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
        
    # 2. Messages table

    # Add app_id and user_id to messages table
    op.add_column("messages", sa.Column("app_id", sa.Text()), schema=schema)
    op.add_column("messages", sa.Column("user_id", sa.Text()), schema=schema)

    # Create foreign key constraints for app_id and user_id
    try:
        op.create_foreign_key(
            "fk_messages_app_id_apps",
            "messages",
            "apps",
            ["app_id"],
            ["public_id"]
        )
        print("Created app_id foreign key")
    except IntegrityError:
        print("Cannot create app_id foreign key - integrity error")
        
    try:
        op.create_foreign_key(
            "fk_messages_user_id_users",
            "messages",
            "users",
            ["user_id"],
            ["public_id"]
        )
        print("Created user_id foreign key")
    except IntegrityError:
        print("Cannot create user_id foreign key - integrity error")

    # Data migration: Fill app_id and user_id from sessions table
    try:
        op.execute("""
            UPDATE messages m
            SET app_id = s.app_id, user_id = s.user_id
            FROM sessions s
            WHERE m.session_id = s.public_id
            AND m.app_id IS NULL
            AND m.user_id IS NULL
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
        
    # 3. Metamessages table
    # Add app_id to metamessages table
    op.add_column("metamessages", sa.Column("app_id", sa.Text()), schema=schema)
    print("Added app_id column to metamessages table")

    # Create foreign key constraint for app_id
    try:
        op.create_foreign_key(
            "fk_metamessages_app_id_apps",
            "metamessages",
            "apps",
            ["app_id"],
            ["public_id"]
        )
        print("Created app_id foreign key")
    except IntegrityError:
        print("Cannot create app_id foreign key - integrity error")
        
    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE metamessages m
            SET app_id = u.app_id
            FROM users u
            WHERE m.user_id = u.public_id
            AND m.app_id IS NULL
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
        
    # 4. Collections table

    # Add app_id to collections table
    op.add_column("collections", sa.Column("app_id", sa.Text()), schema=schema)
    print("Added app_id column to collections table")

    # Create foreign key constraint for app_id
    try:
        op.create_foreign_key(
            "fk_collections_app_id_apps",
            "collections",
            "apps",
            ["app_id"],
            ["public_id"]
        )
        print("Created app_id foreign key for collections table")
    except IntegrityError:
        print("Cannot create app_id foreign key for collections table - integrity error")
        
    # Data migration: Fill app_id from users table
    try:
        op.execute("""
            UPDATE collections c
            SET app_id = u.app_id
            FROM users u
            WHERE c.user_id = u.public_id
            AND c.app_id IS NULL
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
        
    # 5. Documents table

    # Add app_id and user_id to documents table
    op.add_column("documents", sa.Column("app_id", sa.Text()), schema=schema)
    op.add_column("documents", sa.Column("user_id", sa.Text()), schema=schema)
    print("Added app_id and user_id columns to documents table")

    # Create foreign key constraints for app_id and user_id
    try:
        op.create_foreign_key(
            "fk_documents_app_id_apps",
            "documents",
            "apps",
            ["app_id"],
            ["public_id"]
        )
        print("Created app_id foreign key for documents table")
    except IntegrityError:
        print("Cannot create app_id foreign key for documents table - integrity error")
        
    try:
        op.create_foreign_key(
            "fk_documents_user_id_users",
            "documents",
            "users",
            ["user_id"],
            ["public_id"]
        )
        print("Created user_id foreign key for documents table")
    except IntegrityError:
        print("Cannot create user_id foreign key for documents table - integrity error")
            
    # Data migration: Fill app_id and user_id from users table
    try:
        op.execute("""
            UPDATE documents d
            SET app_id = c.app_id, user_id = c.user_id
            FROM collections c
            WHERE d.collection_id = c.public_id
            AND d.app_id IS NULL
            AND d.user_id IS NULL
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

def downgrade():
    schema = getenv("DATABASE_SCHEMA", "public")
    
    # 5. Documents table
    # Make app_id and user_id nullable again
    op.alter_column("documents", "app_id", nullable=True, schema=schema)
    op.alter_column("documents", "user_id", nullable=True, schema=schema)
    print("Made app_id and user_id nullable again for documents table")
    
    # Drop foreign key constraints
    try:
        op.drop_constraint("fk_documents_user_id_users", "documents", schema=schema)
        print("Dropped user_id foreign key for documents table")
    except Exception as e:
        print(f"Error dropping user_id foreign key for documents table: {e}")
        
    try:
        op.drop_constraint("fk_documents_app_id_apps", "documents", schema=schema)
        print("Dropped app_id foreign key for documents table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for documents table: {e}")
    
    # Drop the columns
    op.drop_column("documents", "user_id", schema=schema)
    op.drop_column("documents", "app_id", schema=schema)
    print("Dropped app_id and user_id columns from documents table")
    
    # 4. Collections table
    # Make app_id nullable again
    op.alter_column("collections", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for collections table")
    
    # Drop foreign key constraint
    try:
        op.drop_constraint("fk_collections_app_id_apps", "collections", schema=schema)
        print("Dropped app_id foreign key for collections table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for collections table: {e}")
    
    # Drop the column
    op.drop_column("collections", "app_id", schema=schema)
    print("Dropped app_id column from collections table")
    
    # 3. Metamessages table
    # Make app_id nullable again
    op.alter_column("metamessages", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for metamessages table")
    
    # Drop foreign key constraint
    try:
        op.drop_constraint("fk_metamessages_app_id_apps", "metamessages", schema=schema)
        print("Dropped app_id foreign key for metamessages table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for metamessages table: {e}")
    
    # Drop the column
    op.drop_column("metamessages", "app_id", schema=schema)
    print("Dropped app_id column from metamessages table")
    
    # 2. Messages table
    # Make app_id and user_id nullable again
    op.alter_column("messages", "app_id", nullable=True, schema=schema)
    op.alter_column("messages", "user_id", nullable=True, schema=schema)
    print("Made app_id and user_id nullable again for messages table")
    
    # Drop foreign key constraints
    try:
        op.drop_constraint("fk_messages_user_id_users", "messages", schema=schema)
        print("Dropped user_id foreign key for messages table")
    except Exception as e:
        print(f"Error dropping user_id foreign key for messages table: {e}")
        
    try:
        op.drop_constraint("fk_messages_app_id_apps", "messages", schema=schema)
        print("Dropped app_id foreign key for messages table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for messages table: {e}")
    
    # Drop the columns
    op.drop_column("messages", "user_id", schema=schema)
    op.drop_column("messages", "app_id", schema=schema)
    print("Dropped app_id and user_id columns from messages table")
    
    # 1. Sessions table
    # Make app_id nullable again
    op.alter_column("sessions", "app_id", nullable=True, schema=schema)
    print("Made app_id nullable again for sessions table")
    
    # Drop foreign key constraint
    try:
        op.drop_constraint("fk_sessions_app_id_apps", "sessions", schema=schema)
        print("Dropped app_id foreign key for sessions table")
    except Exception as e:
        print(f"Error dropping app_id foreign key for sessions table: {e}")
    
    # Drop the column
    op.drop_column("sessions", "app_id", schema=schema)
    print("Dropped app_id column from sessions table")