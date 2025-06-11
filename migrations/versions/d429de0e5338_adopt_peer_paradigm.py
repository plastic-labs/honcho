"""adopt peer paradigm

Revision ID: d429de0e5338
Revises: 66e63cf2cf77
Create Date: 2025-06-09 15:16:38.164067

"""
from os import getenv
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from nanoid import generate as generate_nanoid
from sqlalchemy.dialects import postgresql

from migrations.utils import check_constraint_exists, primary_constraint_exists, column_exists, fk_exists, index_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = 'd429de0e5338'
down_revision: Union[str, None] = '66e63cf2cf77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade database schema to adopt peer paradigm."""
    schema = getenv("DATABASE_SCHEMA", "public")
    inspector = sa.inspect(op.get_bind())
    
    # Step 1: Rename tables
    rename_tables(schema, inspector)
    
    # Step 2: Update workspaces table
    update_workspaces_table(schema, inspector)
    
    # Step 3: Update peers table
    update_peers_table(schema, inspector)
    
    # Step 4: Update sessions table
    update_sessions_table(schema, inspector)
    
    # Step 5: Create and populate session_peers table
    create_and_populate_session_peers_table(schema, inspector)
    
    # Step 6: Update messages table
    update_messages_table(schema, inspector) # TODO: finish this

    # Step 7: Update collections table
    update_collections_table(schema, inspector)

    # Step 8: Update documents table
    update_documents_table(schema, inspector)

    # Step 9: Drop metamessages table
    if table_exists("metamessages", inspector):
        op.drop_table("metamessages", schema=schema)


def downgrade() -> None:
    pass

def rename_tables(schema: str, inspector) -> None:
    """Rename apps->workspaces and users->peers tables."""
    if inspector.has_table("apps", schema=schema):
        op.rename_table('apps', 'workspaces', schema=schema)
    if inspector.has_table("users", schema=schema):
        op.rename_table('users', 'peers', schema=schema)


def update_workspaces_table(schema: str, inspector) -> None:
    """Update workspaces table (formerly apps)."""
    
    # Add feature flags column
    op.add_column("workspaces", sa.Column("feature_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}", schema=schema))

    # Update constraint names
    if check_constraint_exists("workspaces", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "workspaces", type_="check", schema=schema)
    if check_constraint_exists("workspaces", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "workspaces", type_="check", schema=schema)

    op.create_check_constraint('id_length', 'workspaces', "length(id) = 21", schema=schema)
    op.create_check_constraint('id_format', 'workspaces', "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)

    # Rename public_id to id and make it the primary key
    if primary_constraint_exists("workspaces", "pk_apps", inspector):
        op.drop_constraint("pk_apps", "workspaces", type_="primary", schema=schema)
    if column_exists("workspaces", "id", inspector):
        op.drop_column("workspaces", "id", schema=schema)

    op.alter_column("workspaces", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_workspaces", "workspaces", ["id"], schema=schema)


def update_peers_table(schema: str, inspector) -> None:
    """Update peers table (formerly users)."""
    
    # Add feature flags column
    op.add_column("peers", sa.Column("feature_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}", schema=schema))

    # Add workspace_name column and migrate data using app_id
    op.add_column("peers", sa.Column("workspace_name", sa.TEXT(), nullable=True, schema=schema))
    op.execute(sa.text(f"UPDATE {schema}.peers SET workspace_name = workspaces.name FROM {schema}.workspaces WHERE peers.app_id = workspaces.id"))    
    
    # Update constraints and indexes
    if check_constraint_exists("peers", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "peers", type_="check", schema=schema)
    if check_constraint_exists("peers", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "peers", type_="check", schema=schema)

    op.create_check_constraint('id_length', 'peers', "length(id) = 21", schema=schema)
    op.create_check_constraint('id_format', 'peers', "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)

    # Update primary key
    if primary_constraint_exists("peers", "pk_users", inspector):
        op.drop_constraint("pk_users", "peers", type_="primary", schema=schema)
    op.drop_column("peers", "id", schema=schema)
    op.alter_column("peers", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_peers", "peers", ["id"], schema=schema)

    # Update foreign key
    if fk_exists("peers", "fk_users_app_id_apps", inspector):
        op.drop_constraint("fk_users_app_id_apps", "peers", type_="foreignkey", schema=schema)
    if fk_exists("peers", "users_app_id_fkey", inspector):
        op.drop_constraint("users_app_id_fkey", "peers", type_="foreignkey", schema=schema)

    op.create_foreign_key("fk_peers_workspace_name_workspaces", "peers", "workspaces", ["workspace_name"], ["name"], schema=schema)
    
    # Update unique constraint
    op.drop_constraint("unique_name_app_user", "peers", type_="unique", schema=schema)
    op.create_unique_constraint("unique_name_workspace_peer", "peers", ["name", "workspace_name"], schema=schema)

    # Update indexes
    op.drop_index("idx_users_app_lookup", "peers", schema=schema)
    op.create_index("idx_peers_workspace_lookup", "peers", ["workspace_name", "name"], schema=schema)


def update_sessions_table(schema: str, inspector) -> None:
    """Update sessions table."""
    
    # Add feature flags column
    op.add_column("sessions", sa.Column("feature_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}", schema=schema))

    # Add workspace_name column and migrate data using app_id
    op.add_column("sessions", sa.Column("workspace_name", sa.TEXT(), nullable=True, schema=schema))
    op.execute(sa.text(f"UPDATE {schema}.sessions SET workspace_name = workspaces.name FROM {schema}.workspaces WHERE sessions.app_id = workspaces.id"))    

    op.add_column('sessions', sa.Column('name', sa.TEXT(), nullable=True, server_default=generate_nanoid(), schema=schema)) # Temporarily nullable
    op.execute(sa.text(f"UPDATE {schema}.sessions SET name = public_id"))
    op.alter_column('sessions', 'name', nullable=False, schema=schema)
    op.create_unique_constraint("unique_session_name", "sessions", ["name", "workspace_name"], schema=schema)

    # Update constraints and indexes
    if check_constraint_exists("sessions", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "sessions", type_="check", schema=schema)
    if check_constraint_exists("sessions", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "sessions", type_="check", schema=schema)

    op.create_check_constraint('id_length', 'sessions', "length(id) = 21", schema=schema)
    op.create_check_constraint('id_format', 'sessions', "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)

    # Update sessions table primary key and foreign keys
    if primary_constraint_exists("sessions", "pk_sessions", inspector):
        op.drop_constraint("pk_sessions", "sessions", type_="primary", schema=schema)

    op.drop_column("sessions", "id", schema=schema)
    op.alter_column("sessions", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_sessions", "sessions", ["id"], schema=schema)

    # Drop old foreign keys
    if fk_exists("sessions", "fk_sessions_user_id_users", inspector):
        op.drop_constraint("fk_sessions_user_id_users", "sessions", type_="foreignkey", schema=schema)
    if fk_exists("sessions", "fk_sessions_app_id_apps", inspector):
        op.drop_constraint("fk_sessions_app_id_apps", "sessions", type_="foreignkey", schema=schema)
    if fk_exists("sessions", "sessions_app_id_fkey", inspector):
        op.drop_constraint("sessions_app_id_fkey", "sessions", type_="foreignkey", schema=schema)

    # Create new foreign key
    op.create_foreign_key("fk_sessions_workspace_name_workspaces", "sessions", "workspaces", ["workspace_name"], ["name"], schema=schema)
    

def create_and_populate_session_peers_table(schema: str, inspector) -> None:
    """Create and populate session_peers table."""
    
    # Create session_peers table
    if not table_exists("session_peers", inspector):
        op.create_table('session_peers',
            sa.Column('session_name', sa.TEXT(), nullable=False),
            sa.Column('peer_name', sa.TEXT(), nullable=False),
            sa.ForeignKeyConstraint(['peer_name'], ['peers.name'], ),
            sa.ForeignKeyConstraint(['session_name'], ['sessions.name'], ),
            sa.PrimaryKeyConstraint('session_name', 'peer_name')
        )

    # Add agent peer for each workspace
    conn = op.get_bind()
    workspaces = conn.execute(sa.text(f"SELECT id, name FROM {schema}.workspaces"))
    agent_peers_map = {}
    for workspace in workspaces:
        workspace_id, workspace_name = workspace
        agent_peer_id = generate_nanoid()
        agent_peers_map[workspace_name] = (agent_peer_id, workspace_id)
        op.execute(
            sa.text(f"INSERT INTO {schema}.peers (id, name, workspace_name) VALUES (:id, :name, :workspace_name)").bindparams(
                id=agent_peer_id,
                name=agent_peer_id,
                workspace_name=workspace_name
            )
        )

    # Populate session_peers table
    sessions = conn.execute(sa.text(f"SELECT user_id, name, workspace_name FROM {schema}.sessions")).fetchall()
    for session in sessions:
        user_id, session_name, workspace_name = session
        op.execute(sa.text("INSERT INTO session_peers (session_name, peer_name) VALUES (:session_name, :peer_name)").bindparams(
            session_name=session_name,
            peer_name=user_id
        ))
        agent_peer_id, _ = agent_peers_map[workspace_name]
        op.execute(
            sa.text(f"INSERT INTO {schema}.session_peers (session_name, peer_name) VALUES (:session_name, :peer_name)").bindparams(
                session_name=session_name,
                peer_name=agent_peer_id
            )
        )


def update_messages_table(schema: str, inspector) -> None:
    """Update messages table."""
    
    # Add new columns
    if not column_exists("messages", "peer_name", inspector):
        op.add_column("messages", sa.Column("peer_name", sa.TEXT(), nullable=True, schema=schema))
    if not column_exists("messages", "workspace_name", inspector):
        op.add_column("messages", sa.Column("workspace_name", sa.TEXT(), nullable=True, schema=schema))
    if not column_exists("messages", "session_name", inspector):
        op.add_column("messages", sa.Column("session_name", sa.TEXT(), nullable=True, schema=schema))
    
    op.execute(sa.text(f"""
        UPDATE {schema}.messages SET 
        workspace_name = (SELECT workspace_name FROM {schema}.workspaces WHERE workspaces.id = messages.app_id)
    """))
    
    op.execute(sa.text(f"""
        UPDATE {schema}.messages SET 
        session_name = (SELECT name FROM {schema}.sessions WHERE sessions.id = messages.session_id)
    """))

    op.execute(sa.text(f"""
        UPDATE {schema}.messages SET
        peer_name = CASE 
            WHEN is_user = true THEN (
                SELECT p.name 
                FROM {schema}.peers p 
                JOIN {schema}.sessions s ON s.user_id = p.id 
                WHERE s.name = messages.session_name
            )
            ELSE (
                SELECT sp.peer_name 
                FROM {schema}.session_peers sp 
                WHERE sp.session_name = messages.session_name 
                AND sp.peer_name != (
                    SELECT p.name 
                    FROM {schema}.peers p 
                    JOIN {schema}.sessions s ON s.user_id = p.id 
                    WHERE s.name = messages.session_name
                )
            )
        END
    """))

    # Make columns not nullable
    op.alter_column("messages", "peer_name", nullable=False, schema=schema)
    op.alter_column("messages", "workspace_name", nullable=False, schema=schema)
    op.alter_column("messages", "session_name", nullable=False, schema=schema)
    
    # Drop old columns and constraints
    if fk_exists("messages", "fk_messages_session_id_sessions", inspector):
        op.drop_constraint("fk_messages_session_id_sessions", "messages", type_="foreignkey", schema=schema)
    if fk_exists("messages", "messages_app_id_fkey", inspector):
        op.drop_constraint("messages_app_id_fkey", "messages", type_="foreignkey", schema=schema)
    if fk_exists("messages", "fk_messages_app_id_apps", inspector):
        op.drop_constraint("fk_messages_app_id_apps", "messages", type_="foreignkey", schema=schema)
    if fk_exists("messages", "messages_user_id_fkey", inspector):
        op.drop_constraint("messages_user_id_fkey", "messages", type_="foreignkey", schema=schema)
    if fk_exists("messages", "fk_messages_user_id_users", inspector):
        op.drop_constraint("fk_messages_user_id_users", "messages", type_="foreignkey", schema=schema)
    
    op.drop_column("messages", "is_user", schema=schema)
    op.drop_column("messages", "user_id", schema=schema)
    op.drop_column("messages", "app_id", schema=schema)
    op.drop_column("messages", "session_id", schema=schema)
    
    # Add new foreign keys
    op.create_foreign_key("fk_messages_session_name_sessions", "messages", "sessions", ["session_name"], ["name"], schema=schema)
    op.create_foreign_key("fk_messages_peer_name_peers", "messages", "peers", ["peer_name"], ["name"], schema=schema)
    op.create_foreign_key("fk_messages_workspace_name_workspaces", "messages", "workspaces", ["workspace_name"], ["name"], schema=schema)
    
    # Update indexes
    op.drop_index("idx_messages_session_lookup", table_name="messages", schema=schema)
    op.create_index("idx_messages_session_lookup", "messages", ["session_name", "id"], 
                   postgresql_include=["id", "created_at"], schema=schema)
    op.create_index("ix_messages_peer_name", "messages", ["peer_name"], schema=schema)
    op.create_index("ix_messages_workspace_name", "messages", ["workspace_name"], schema=schema)
    
    # Update constraints
    if check_constraint_exists("messages", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "messages", type_="check", schema=schema)
    if check_constraint_exists("messages", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "messages", type_="check", schema=schema)
    
    op.create_check_constraint("id_length", "messages", "length(id) = 21", schema=schema)
    op.create_check_constraint("id_format", "messages", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)

def update_collections_table(schema: str, inspector) -> None:
    """Update collections table."""
    
    # Add new columns
    if not column_exists("collections", "peer_name", inspector):
        op.add_column("collections", sa.Column("peer_name", sa.TEXT(), nullable=True, schema=schema))
    if not column_exists("collections", "workspace_name", inspector):
        op.add_column("collections", sa.Column("workspace_name", sa.TEXT(), nullable=True, schema=schema))
    
    # Populate new columns from existing data
    op.execute(sa.text(f"""
        UPDATE {schema}.collections SET 
        peer_name = (SELECT name FROM {schema}.peers WHERE peers.id = collections.user_id),
        workspace_name = (SELECT workspace_name FROM {schema}.workspaces WHERE workspaces.id = collections.app_id)
    """))
    
    # Make columns not nullable
    op.alter_column("collections", "peer_name", nullable=False, schema=schema)
    op.alter_column("collections", "workspace_name", nullable=False, schema=schema)
    
    # Update primary key structure
    if primary_constraint_exists("collections", "pk_collections", inspector):
        op.drop_constraint("pk_collections", "collections", type_="primary", schema=schema)
    if column_exists("collections", "id", inspector):
        op.drop_column("collections", "id", schema=schema)
    op.alter_column("collections", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_collections", "collections", ["id"], schema=schema)
    
    # Drop old constraints and columns
    if fk_exists("collections", "fk_collections_user_id_users", inspector):
        op.drop_constraint("fk_collections_user_id_users", "collections", type_="foreignkey", schema=schema)
    if fk_exists("collections", "fk_collections_app_id_apps", inspector):
        op.drop_constraint("fk_collections_app_id_apps", "collections", type_="foreignkey", schema=schema)
    if fk_exists("collections", "collections_app_id_fkey", inspector):
        op.drop_constraint("collections_app_id_fkey", "collections", type_="foreignkey", schema=schema)
    if fk_exists("collections", "collections_user_id_fkey", inspector):
        op.drop_constraint("collections_user_id_fkey", "collections", type_="foreignkey", schema=schema)
    
    op.drop_constraint("unique_name_collection_user", "collections", type_="unique", schema=schema)
    op.drop_column("collections", "user_id", schema=schema)
    op.drop_column("collections", "app_id", schema=schema)
    
    # Add new constraints
    op.create_unique_constraint("unique_name_collection_peer", "collections", ["name", "peer_name"], schema=schema)
    op.create_foreign_key("fk_collections_peer_name_peers", "collections", "peers", ["peer_name"], ["name"], schema=schema)
    op.create_foreign_key("fk_collections_workspace_name_workspaces", "collections", "workspaces", ["workspace_name"], ["name"], schema=schema)
    
    # Update constraint names
    if check_constraint_exists("collections", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "collections", type_="check", schema=schema)
    if check_constraint_exists("collections", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "collections", type_="check", schema=schema)
    
    op.create_check_constraint("id_length", "collections", "length(id) = 21", schema=schema)
    op.create_check_constraint("id_format", "collections", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)
    op.create_check_constraint("name_length", "collections", "length(name) <= 512", schema=schema)

def update_documents_table(schema: str, inspector) -> None:
    """Update documents table."""
    
    # Add new columns
    if not column_exists("documents", "peer_name", inspector):
        op.add_column("documents", sa.Column("peer_name", sa.TEXT(), nullable=True, schema=schema))
    if not column_exists("documents", "workspace_name", inspector):
        op.add_column("documents", sa.Column("workspace_name", sa.TEXT(), nullable=True, schema=schema))
    
    # Populate new columns from existing data
    op.execute(sa.text(f"""
        UPDATE {schema}.documents SET 
        peer_name = (SELECT name FROM {schema}.peers WHERE peers.id = documents.user_id),
        workspace_name = (SELECT workspace_name FROM {schema}.workspaces WHERE workspaces.id = documents.app_id)
    """))
    
    # Make columns not nullable
    op.alter_column("documents", "peer_name", nullable=False, schema=schema)
    op.alter_column("documents", "workspace_name", nullable=False, schema=schema)
    
    # Update primary key structure
    if primary_constraint_exists("documents", "pk_documents", inspector):
        op.drop_constraint("pk_documents", "documents", type_="primary", schema=schema)
    if column_exists("documents", "id", inspector):
        op.drop_column("documents", "id", schema=schema)
    op.alter_column("documents", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_documents", "documents", ["id"], schema=schema)
    
    # Update collection reference
    op.alter_column("documents", "collection_id", new_column_name="collection_name", schema=schema)
    
    # Drop old constraints and columns
    if fk_exists("documents", "fk_documents_collection_id_collections", inspector):
        op.drop_constraint("fk_documents_collection_id_collections", "documents", type_="foreignkey", schema=schema)
    if fk_exists("documents", "fk_documents_user_id_users", inspector):
        op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey", schema=schema)
    if fk_exists("documents", "fk_documents_app_id_apps", inspector):
        op.drop_constraint("fk_documents_app_id_apps", "documents", type_="foreignkey", schema=schema)
    if fk_exists("documents", "documents_app_id_fkey", inspector):
        op.drop_constraint("documents_app_id_fkey", "documents", type_="foreignkey", schema=schema)
    if fk_exists("documents", "documents_user_id_fkey", inspector):
        op.drop_constraint("documents_user_id_fkey", "documents", type_="foreignkey", schema=schema)
    
    op.drop_column("documents", "user_id", schema=schema)
    op.drop_column("documents", "app_id", schema=schema)
    
    # Add new foreign keys
    op.create_foreign_key("fk_documents_collection_name_collections", "documents", "collections", ["collection_name"], ["name"], schema=schema)
    op.create_foreign_key("fk_documents_peer_name_peers", "documents", "peers", ["peer_name"], ["name"], schema=schema)
    op.create_foreign_key("fk_documents_workspace_name_workspaces", "documents", "workspaces", ["workspace_name"], ["name"], schema=schema)
    
    # Update constraint names
    if check_constraint_exists("documents", "public_id_length", inspector):
        op.drop_constraint("public_id_length", "documents", type_="check", schema=schema)
    if check_constraint_exists("documents", "public_id_format", inspector):
        op.drop_constraint("public_id_format", "documents", type_="check", schema=schema)
    
    op.create_check_constraint("id_length", "documents", "length(id) = 21", schema=schema)
    op.create_check_constraint("id_format", "documents", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema)
    