"""adopt peer paradigm

Revision ID: d429de0e5338
Revises: 66e63cf2cf77
Create Date: 2025-06-09 15:16:38.164067

"""

from collections.abc import Sequence
from contextlib import suppress
from os import getenv
from typing import Union

import sqlalchemy as sa
import tiktoken
from alembic import op
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from migrations.utils import (
    column_exists,
    constraint_exists,
    fk_exists,
    index_exists,
    table_exists,
)

# revision identifiers, used by Alembic.
revision: str = "d429de0e5338"
down_revision: Union[str, None] = "66e63cf2cf77"
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

    # Step 4: Update queue and active_queue_sessions tables (moved before sessions update)
    update_queue_and_active_queue_sessions_tables(schema, inspector)

    # Step 5: Update sessions table
    update_sessions_table(schema, inspector)

    # Step 6: Create and populate session_peers table
    create_and_populate_session_peers_table(schema, inspector)

    # Step 7: Update messages table
    update_messages_table(schema, inspector)

    # Step 8: Update collections table
    update_collections_table(schema, inspector)

    # Step 9: Update documents table
    update_documents_table(schema, inspector)

    # Step 10: Drop metamessages table
    if table_exists("metamessages", inspector):
        op.drop_table("metamessages", schema=schema)

    # Step 11: Drop app_id, user_id from peers and sessions
    if column_exists("sessions", "app_id", inspector):
        op.drop_column("sessions", "app_id", schema=schema)
    if column_exists("sessions", "user_id", inspector):
        op.drop_column("sessions", "user_id", schema=schema)


def downgrade() -> None:
    """Downgrade database schema to reverse peer paradigm adoption."""
    schema = getenv("DATABASE_SCHEMA", "public")
    inspector = sa.inspect(op.get_bind())

    # Step 1: Add back app_id, user_id to peers and sessions
    restore_app_user_columns(schema, inspector)

    # Step 2: Restore documents table
    restore_documents_table(schema, inspector)

    # Step 3: Restore collections table
    restore_collections_table(schema, inspector)

    # Step 4: Restore messages table
    restore_messages_table(schema, inspector)

    # Step 5: Drop session_peers table
    if table_exists("session_peers", inspector):
        op.drop_table("session_peers", schema=schema)

    # Step 6: Restore sessions table
    restore_sessions_table(schema, inspector)

    # Step 7: Restore peers table
    restore_peers_table(schema, inspector)

    # Step 8: Restore workspaces table
    restore_workspaces_table(schema, inspector)

    # Step 9: Restore queue and active_queue_sessions tables
    restore_queue_and_active_queue_sessions_tables(schema, inspector)

    # Step 10: Rename tables back
    restore_table_names(schema, inspector)

    # Step 11: Restore foreign keys
    restore_foreign_keys(schema)


def rename_tables(schema: str, inspector) -> None:
    """Rename apps->workspaces and users->peers tables."""
    if inspector.has_table("apps", schema=schema):
        op.rename_table("apps", "workspaces", schema=schema)
    if inspector.has_table("users", schema=schema):
        op.rename_table("users", "peers", schema=schema)


def update_workspaces_table(schema: str, inspector) -> None:
    """Update workspaces table (formerly apps)."""

    # Add configuration column
    op.add_column(
        "workspaces",
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )
    # Add internal_metadata column
    op.add_column(
        "workspaces",
        sa.Column(
            "internal_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Update constraint names
    if constraint_exists("workspaces", "public_id_length", "check", inspector):
        op.drop_constraint(
            "public_id_length", "workspaces", type_="check", schema=schema
        )
    if constraint_exists("workspaces", "public_id_format", "check", inspector):
        op.drop_constraint(
            "public_id_format", "workspaces", type_="check", schema=schema
        )

    # Rename public_id to id and make it the primary key
    if constraint_exists("workspaces", "pk_apps", "primary", inspector):
        op.drop_constraint("pk_apps", "workspaces", type_="primary", schema=schema)
    if column_exists("workspaces", "id", inspector):
        op.drop_column("workspaces", "id", schema=schema)

    op.alter_column("workspaces", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_workspaces", "workspaces", ["id"], schema=schema)

    op.create_check_constraint(
        "id_length", "workspaces", "length(id) = 21", schema=schema
    )
    op.create_check_constraint(
        "id_format", "workspaces", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def update_peers_table(schema: str, inspector) -> None:
    """Update peers table (formerly users)."""

    # Add configuration column
    op.add_column(
        "peers",
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Add internal_metadata column
    op.add_column(
        "peers",
        sa.Column(
            "internal_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Add workspace_name column and migrate data using app_id
    op.add_column(
        "peers", sa.Column("workspace_name", sa.TEXT(), nullable=True), schema=schema
    )
    op.execute(
        sa.text(
            f"UPDATE {schema}.peers SET workspace_name = workspaces.name FROM {schema}.workspaces WHERE peers.app_id = workspaces.id"
        )
    )

    # Update constraints and indexes
    if constraint_exists("peers", "public_id_length", "check", inspector):
        op.drop_constraint("public_id_length", "peers", type_="check", schema=schema)
    if constraint_exists("peers", "public_id_format", "check", inspector):
        op.drop_constraint("public_id_format", "peers", type_="check", schema=schema)

    # Update primary key
    if constraint_exists("peers", "pk_users", "primary", inspector):
        op.drop_constraint("pk_users", "peers", type_="primary", schema=schema)
    op.drop_column("peers", "id", schema=schema)
    op.alter_column("peers", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_peers", "peers", ["id"], schema=schema)

    # Update foreign key
    if fk_exists("peers", "fk_users_app_id_apps", inspector):
        op.drop_constraint(
            "fk_users_app_id_apps", "peers", type_="foreignkey", schema=schema
        )
    if fk_exists("peers", "users_app_id_fkey", inspector):
        op.drop_constraint(
            "users_app_id_fkey", "peers", type_="foreignkey", schema=schema
        )

    op.create_foreign_key(
        "fk_peers_workspace_name_workspaces",
        "peers",
        "workspaces",
        ["workspace_name"],
        ["name"],
        referent_schema=schema,
    )

    # Update unique constraint
    if constraint_exists("peers", "unique_name_app_user", "unique", inspector):
        op.drop_constraint(
            "unique_name_app_user", "peers", type_="unique", schema=schema
        )
    op.create_unique_constraint(
        "unique_name_workspace_peer", "peers", ["name", "workspace_name"], schema=schema
    )

    # Update indexes
    op.drop_index("idx_users_app_lookup", "peers", schema=schema)
    op.create_index(
        "idx_peers_workspace_lookup", "peers", ["workspace_name", "name"], schema=schema
    )

    op.create_check_constraint("id_length", "peers", "length(id) = 21", schema=schema)
    op.create_check_constraint(
        "id_format", "peers", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )

    op.drop_column("peers", "app_id", schema=schema)


def update_sessions_table(schema: str, inspector) -> None:
    """Update sessions table."""

    # Add configuration column
    op.add_column(
        "sessions",
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Add internal_metadata column
    op.add_column(
        "sessions",
        sa.Column(
            "internal_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Add workspace_name column and migrate data using app_id
    op.add_column(
        "sessions", sa.Column("workspace_name", sa.TEXT(), nullable=True), schema=schema
    )
    op.execute(
        sa.text(
            f"UPDATE {schema}.sessions SET workspace_name = workspaces.name FROM {schema}.workspaces WHERE sessions.app_id = workspaces.id"
        )
    )

    op.add_column(
        "sessions",
        sa.Column("name", sa.TEXT(), nullable=True),
        schema=schema,
    )  # Temporarily nullable
    op.execute(sa.text(f"UPDATE {schema}.sessions SET name = public_id"))
    op.alter_column("sessions", "name", nullable=False, schema=schema)
    op.create_unique_constraint(
        "unique_session_name", "sessions", ["name", "workspace_name"], schema=schema
    )

    # Update constraints and indexes
    if constraint_exists("sessions", "public_id_length", "check", inspector):
        op.drop_constraint("public_id_length", "sessions", type_="check", schema=schema)
    if constraint_exists("sessions", "public_id_format", "check", inspector):
        op.drop_constraint("public_id_format", "sessions", type_="check", schema=schema)

    # Update sessions table primary key and foreign keys
    if constraint_exists("sessions", "pk_sessions", "primary", inspector):
        op.drop_constraint("pk_sessions", "sessions", type_="primary", schema=schema)

    op.drop_column("sessions", "id", schema=schema)
    op.alter_column("sessions", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_sessions", "sessions", ["id"], schema=schema)

    # Drop old foreign keys
    if fk_exists("sessions", "fk_sessions_user_id_users", inspector):
        op.drop_constraint(
            "fk_sessions_user_id_users", "sessions", type_="foreignkey", schema=schema
        )
    if fk_exists("sessions", "fk_sessions_app_id_apps", inspector):
        op.drop_constraint(
            "fk_sessions_app_id_apps", "sessions", type_="foreignkey", schema=schema
        )
    if fk_exists("sessions", "sessions_app_id_fkey", inspector):
        op.drop_constraint(
            "sessions_app_id_fkey", "sessions", type_="foreignkey", schema=schema
        )

    # Create new foreign key
    op.create_foreign_key(
        "fk_sessions_workspace_name_workspaces",
        "sessions",
        "workspaces",
        ["workspace_name"],
        ["name"],
        referent_schema=schema,
    )

    op.create_check_constraint(
        "id_length", "sessions", "length(id) = 21", schema=schema
    )
    op.create_check_constraint(
        "id_format", "sessions", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )
    op.create_check_constraint(
        "name_length", "sessions", "length(name) <= 512", schema=schema
    )


def create_and_populate_session_peers_table(schema: str, inspector) -> None:
    """Create and populate session_peers table."""

    # Create session_peers table
    if not table_exists("session_peers", inspector):
        op.create_table(
            "session_peers",
            sa.Column("workspace_name", sa.TEXT(), nullable=False),
            sa.Column("session_name", sa.TEXT(), nullable=False),
            sa.Column("peer_name", sa.TEXT(), nullable=False),
            sa.Column(
                "configuration",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default="{}",
            ),
            sa.Column(
                "internal_metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default="{}",
            ),
            sa.Column(
                "joined_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "left_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(
                ["peer_name", "workspace_name"],
                ["peers.name", "peers.workspace_name"],
            ),
            sa.ForeignKeyConstraint(
                ["session_name", "workspace_name"],
                ["sessions.name", "sessions.workspace_name"],
            ),
            sa.PrimaryKeyConstraint("workspace_name", "session_name", "peer_name"),
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
            sa.text(
                f"INSERT INTO {schema}.peers (id, name, workspace_name) VALUES (:id, :name, :workspace_name)"
            ).bindparams(
                id=agent_peer_id, name=agent_peer_id, workspace_name=workspace_name
            )
        )

    # Populate session_peers table
    sessions = conn.execute(
        sa.text(f"""
        SELECT s.user_id, s.name, s.workspace_name, p.name as peer_name
        FROM {schema}.sessions s
        JOIN {schema}.peers p ON p.id = s.user_id AND p.workspace_name = s.workspace_name
    """)
    ).fetchall()
    for session in sessions:
        user_id, session_name, workspace_name, peer_name = session
        op.execute(
            sa.text(
                "INSERT INTO session_peers (workspace_name, session_name, peer_name) VALUES (:workspace_name, :session_name, :peer_name)"
            ).bindparams(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=peer_name,
            )
        )
        agent_peer_id, _ = agent_peers_map[workspace_name]
        op.execute(
            sa.text(
                f"INSERT INTO {schema}.session_peers (workspace_name, session_name, peer_name) VALUES (:workspace_name, :session_name, :peer_name)"
            ).bindparams(
                workspace_name=workspace_name,
                session_name=session_name,
                peer_name=agent_peer_id,
            )
        )


def update_messages_table(schema: str, inspector) -> None:
    """Update messages table."""

    # Add new columns
    if not column_exists("messages", "peer_name", inspector):
        op.add_column(
            "messages", sa.Column("peer_name", sa.TEXT(), nullable=True), schema=schema
        )
    if not column_exists("messages", "workspace_name", inspector):
        op.add_column(
            "messages",
            sa.Column("workspace_name", sa.TEXT(), nullable=True),
            schema=schema,
        )
    if not column_exists("messages", "session_name", inspector):
        op.add_column(
            "messages",
            sa.Column("session_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    op.execute(
        sa.text(f"""
        UPDATE {schema}.messages SET 
        workspace_name = (SELECT name FROM {schema}.workspaces WHERE workspaces.id = messages.app_id)
    """)
    )

    op.execute(
        sa.text(f"""
        UPDATE {schema}.messages SET 
        session_name = (SELECT name FROM {schema}.sessions WHERE sessions.id = messages.session_id)
    """)
    )

    op.execute(
        sa.text(f"""
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
    """)
    )

    # Make columns not nullable
    op.alter_column("messages", "peer_name", nullable=False, schema=schema)
    op.alter_column("messages", "workspace_name", nullable=False, schema=schema)

    # Drop old columns and constraints
    if fk_exists("messages", "fk_messages_session_id_sessions", inspector):
        op.drop_constraint(
            "fk_messages_session_id_sessions",
            "messages",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("messages", "messages_app_id_fkey", inspector):
        op.drop_constraint(
            "messages_app_id_fkey", "messages", type_="foreignkey", schema=schema
        )
    if fk_exists("messages", "fk_messages_app_id_apps", inspector):
        op.drop_constraint(
            "fk_messages_app_id_apps", "messages", type_="foreignkey", schema=schema
        )
    if fk_exists("messages", "messages_user_id_fkey", inspector):
        op.drop_constraint(
            "messages_user_id_fkey", "messages", type_="foreignkey", schema=schema
        )

    op.drop_column("messages", "is_user", schema=schema)
    op.drop_column("messages", "user_id", schema=schema)
    op.drop_column("messages", "app_id", schema=schema)
    op.drop_column("messages", "session_id", schema=schema)

    # Add new foreign keys
    op.create_foreign_key(
        "fk_messages_session_name_sessions",
        "messages",
        "sessions",
        ["session_name", "workspace_name"],
        ["name", "workspace_name"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_messages_peer_name_peers",
        "messages",
        "peers",
        ["peer_name", "workspace_name"],
        ["name", "workspace_name"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_messages_workspace_name_workspaces",
        "messages",
        "workspaces",
        ["workspace_name"],
        ["name"],
        referent_schema=schema,
    )

    # Update indexes
    if index_exists("messages", "idx_messages_session_lookup", inspector):
        op.drop_index(
            "idx_messages_session_lookup", table_name="messages", schema=schema
        )
    op.create_index(
        "idx_messages_session_lookup",
        "messages",
        ["session_name", "id"],
        postgresql_include=["id", "created_at"],
        schema=schema,
    )
    op.create_index("ix_messages_peer_name", "messages", ["peer_name"], schema=schema)
    op.create_index(
        "ix_messages_workspace_name", "messages", ["workspace_name"], schema=schema
    )

    # Create full text search index on content column
    op.create_index(
        "idx_messages_content_gin",
        "messages",
        [sa.text("to_tsvector('english', content)")],
        postgresql_using="gin",
        schema=schema,
    )

    op.add_column(
        "messages",
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        schema=schema,
    )
    op.add_column(
        "messages",
        sa.Column(
            "internal_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema=schema,
    )

    # Backfill token counts for existing messages
    backfill_token_counts(schema)


def update_collections_table(schema: str, inspector) -> None:
    """Update collections table."""

    # Add new columns
    if not column_exists("collections", "peer_name", inspector):
        op.add_column(
            "collections",
            sa.Column("peer_name", sa.TEXT(), nullable=True),
            schema=schema,
        )
    if not column_exists("collections", "workspace_name", inspector):
        op.add_column(
            "collections",
            sa.Column("workspace_name", sa.TEXT(), nullable=True),
            schema=schema,
        )
    if not column_exists("collections", "internal_metadata", inspector):
        op.add_column(
            "collections",
            sa.Column(
                "internal_metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default="{}",
            ),
            schema=schema,
        )

    # Populate new columns from existing data
    op.execute(
        sa.text(f"""
        UPDATE {schema}.collections SET 
        peer_name = (SELECT name FROM {schema}.peers WHERE peers.id = collections.user_id),
        workspace_name = (SELECT name FROM {schema}.workspaces WHERE workspaces.id = collections.app_id)
    """)
    )

    # Make columns not nullable
    op.alter_column("collections", "peer_name", nullable=False, schema=schema)
    op.alter_column("collections", "workspace_name", nullable=False, schema=schema)

    # Update primary key structure
    if constraint_exists("collections", "pk_collections", "primary", inspector):
        op.drop_constraint(
            "pk_collections", "collections", type_="primary", schema=schema
        )
    if column_exists("collections", "id", inspector):
        op.drop_column("collections", "id", schema=schema)
    op.alter_column("collections", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_collections", "collections", ["id"], schema=schema)

    # Drop old constraints and columns
    if fk_exists("collections", "fk_collections_user_id_users", inspector):
        op.drop_constraint(
            "fk_collections_user_id_users",
            "collections",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("collections", "fk_collections_app_id_apps", inspector):
        op.drop_constraint(
            "fk_collections_app_id_apps",
            "collections",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("collections", "collections_app_id_fkey", inspector):
        op.drop_constraint(
            "collections_app_id_fkey", "collections", type_="foreignkey", schema=schema
        )
    if fk_exists("collections", "collections_user_id_fkey", inspector):
        op.drop_constraint(
            "collections_user_id_fkey", "collections", type_="foreignkey", schema=schema
        )

    op.drop_constraint(
        "unique_name_collection_user", "collections", type_="unique", schema=schema
    )
    op.drop_column("collections", "user_id", schema=schema)
    op.drop_column("collections", "app_id", schema=schema)

    # Add new constraints
    op.create_unique_constraint(
        "unique_name_collection_peer",
        "collections",
        ["name", "peer_name", "workspace_name"],
        schema=schema,
    )
    op.create_foreign_key(
        "fk_collections_peer_name_peers",
        "collections",
        "peers",
        ["peer_name", "workspace_name"],
        ["name", "workspace_name"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_collections_workspace_name_workspaces",
        "collections",
        "workspaces",
        ["workspace_name"],
        ["name"],
        referent_schema=schema,
    )

    # Update constraint names
    if constraint_exists("collections", "public_id_length", "check", inspector):
        op.drop_constraint(
            "public_id_length", "collections", type_="check", schema=schema
        )
    if constraint_exists("collections", "public_id_format", "check", inspector):
        op.drop_constraint(
            "public_id_format", "collections", type_="check", schema=schema
        )

    op.create_check_constraint(
        "id_length", "collections", "length(id) = 21", schema=schema
    )
    op.create_check_constraint(
        "id_format", "collections", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def update_documents_table(schema: str, inspector) -> None:
    """Update documents table."""

    # Add new columns
    if not column_exists("documents", "peer_name", inspector):
        op.add_column(
            "documents", sa.Column("peer_name", sa.TEXT(), nullable=True), schema=schema
        )
    if not column_exists("documents", "workspace_name", inspector):
        op.add_column(
            "documents",
            sa.Column("workspace_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Populate new columns from existing data
    op.execute(
        sa.text(f"""
        UPDATE {schema}.documents SET 
        peer_name = (SELECT name FROM {schema}.peers WHERE peers.id = documents.user_id),
        workspace_name = (SELECT name FROM {schema}.workspaces WHERE workspaces.id = documents.app_id)
    """)
    )

    # Drop old constraints and columns
    if fk_exists("documents", "fk_documents_collection_id_collections", inspector):
        op.drop_constraint(
            "fk_documents_collection_id_collections",
            "documents",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("documents", "fk_documents_user_id_users", inspector):
        op.drop_constraint(
            "fk_documents_user_id_users", "documents", type_="foreignkey", schema=schema
        )
    if fk_exists("documents", "fk_documents_app_id_apps", inspector):
        op.drop_constraint(
            "fk_documents_app_id_apps", "documents", type_="foreignkey", schema=schema
        )
    if fk_exists("documents", "documents_app_id_fkey", inspector):
        op.drop_constraint(
            "documents_app_id_fkey", "documents", type_="foreignkey", schema=schema
        )
    if fk_exists("documents", "documents_user_id_fkey", inspector):
        op.drop_constraint(
            "documents_user_id_fkey", "documents", type_="foreignkey", schema=schema
        )

    # Now rename the column
    if column_exists("documents", "collection_id", inspector):
        op.alter_column(
            "documents",
            "collection_id",
            new_column_name="collection_name",
            schema=schema,
        )

    # rename metadata to internal_metadata
    op.alter_column(
        "documents",
        "metadata",
        new_column_name="internal_metadata",
        schema=schema,
    )

    # Convert collection_id references to collection names
    # (collection_id contains old collection IDs, we need to get the collection names)
    op.execute(
        sa.text(f"""
        UPDATE {schema}.documents 
        SET collection_name = (
            SELECT c.name 
            FROM {schema}.collections c 
            WHERE c.id = documents.collection_name 
            AND c.peer_name = documents.peer_name 
            AND c.workspace_name = documents.workspace_name
        )
    """)
    )

    # Make columns not nullable
    op.alter_column("documents", "peer_name", nullable=False, schema=schema)
    op.alter_column("documents", "workspace_name", nullable=False, schema=schema)

    # Update primary key structure
    if constraint_exists("documents", "pk_documents", "primary", inspector):
        op.drop_constraint("pk_documents", "documents", type_="primary", schema=schema)
    if column_exists("documents", "id", inspector):
        op.drop_column("documents", "id", schema=schema)
    op.alter_column("documents", "public_id", new_column_name="id", schema=schema)
    op.create_primary_key("pk_documents", "documents", ["id"], schema=schema)

    op.drop_column("documents", "user_id", schema=schema)
    op.drop_column("documents", "app_id", schema=schema)

    # Add new foreign keys
    op.create_foreign_key(
        "fk_documents_collection_name_collections",
        "documents",
        "collections",
        ["collection_name", "peer_name", "workspace_name"],
        ["name", "peer_name", "workspace_name"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_documents_workspace_name_workspaces",
        "documents",
        "workspaces",
        ["workspace_name"],
        ["name"],
        referent_schema=schema,
    )

    # Update constraint names
    if constraint_exists("documents", "public_id_length", "check", inspector):
        op.drop_constraint(
            "public_id_length", "documents", type_="check", schema=schema
        )
    if constraint_exists("documents", "public_id_format", "check", inspector):
        op.drop_constraint(
            "public_id_format", "documents", type_="check", schema=schema
        )

    op.create_check_constraint(
        "id_length", "documents", "length(id) = 21", schema=schema
    )
    op.create_check_constraint(
        "id_format", "documents", "id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def update_queue_and_active_queue_sessions_tables(schema: str, inspector) -> None:
    """Update queue and active_queue_sessions tables."""

    # Drop foreign key constraints first, before changing column types
    if table_exists("queue", inspector) and fk_exists(
        "queue", "fk_queue_session_id_sessions", inspector
    ):
        op.drop_constraint(
            "fk_queue_session_id_sessions",
            "queue",
            type_="foreignkey",
            schema=schema,
        )

    if table_exists("active_queue_sessions", inspector) and fk_exists(
        "active_queue_sessions",
        "fk_active_queue_sessions_session_id_sessions",
        inspector,
    ):
        op.drop_constraint(
            "fk_active_queue_sessions_session_id_sessions",
            "active_queue_sessions",
            type_="foreignkey",
            schema=schema,
        )

    connection = op.get_bind()

    # Get the mapping of old session.id (integer) to new session.id (text, which is public_id)
    # At this point, sessions table still has both id (integer) and public_id (text) columns
    session_id_mapping = {}
    if table_exists("sessions", inspector):
        sessions_mapping = connection.execute(
            sa.text(f"SELECT id, public_id FROM {schema}.sessions")
        ).fetchall()

        for old_id, new_id in sessions_mapping:
            session_id_mapping[old_id] = new_id

    # Update queue table
    if table_exists("queue", inspector):
        # Get current session_id values in queue table
        queue_session_ids = connection.execute(
            sa.text(
                f"SELECT DISTINCT session_id FROM {schema}.queue WHERE session_id IS NOT NULL"
            )
        ).fetchall()

        op.alter_column(
            "queue",
            "session_id",
            type_=sa.Text(),
            existing_type=sa.BigInteger(),
            postgresql_using="session_id::text",
            nullable=True,
        )
        # Convert session_id values in queue table
        for (session_id,) in queue_session_ids:
            if session_id in session_id_mapping:
                new_id = session_id_mapping[session_id]
                connection.execute(
                    sa.text(
                        f"UPDATE {schema}.queue SET session_id = :new_id WHERE session_id = :old_id"
                    ),
                    {"new_id": str(new_id), "old_id": str(session_id)},
                )

    # Update active_queue_sessions table
    if table_exists("active_queue_sessions", inspector):
        active_queue_session_ids = connection.execute(
            sa.text(
                f"SELECT DISTINCT session_id FROM {schema}.active_queue_sessions WHERE session_id IS NOT NULL"
            )
        ).fetchall()

        if constraint_exists(
            "active_queue_sessions", "pk_active_queue_sessions", "primary", inspector
        ):
            op.drop_constraint(
                "pk_active_queue_sessions",
                "active_queue_sessions",
                type_="primary",
                schema=schema,
            )

        op.add_column(
            "active_queue_sessions",
            sa.Column(
                "id",
                sa.TEXT(),
                nullable=True,
            ),
            schema=schema,
        )

        # Update existing rows with unique nanoids
        connection = op.get_bind()

        # Get all rows that need IDs (using session_id as unique identifier)
        rows_needing_ids = connection.execute(
            sa.text(
                f"SELECT session_id FROM {schema}.active_queue_sessions WHERE id IS NULL"
            )
        ).fetchall()

        if rows_needing_ids:
            # Generate nanoids for all rows upfront
            updates = [
                {"session_id": row[0], "nanoid": generate_nanoid()}
                for row in rows_needing_ids
            ]

            # Batch update using individual queries (still better than while loop)
            for update in updates:
                connection.execute(
                    sa.text(
                        f"UPDATE {schema}.active_queue_sessions SET id = :nanoid WHERE session_id = :session_id AND id IS NULL"
                    ),
                    update,
                )

        # Make the column non-nullable after populating data
        op.alter_column("active_queue_sessions", "id", nullable=False, schema=schema)

        op.create_primary_key(
            "pk_active_queue_sessions",
            "active_queue_sessions",
            ["id"],
            schema=schema,
        )

        op.alter_column(
            "active_queue_sessions",
            "session_id",
            type_=sa.Text(),
            nullable=True,
            existing_type=sa.BigInteger(),
            postgresql_using="session_id::text",
        )
        # Convert session_id values in active_queue_sessions table
        for (session_id,) in active_queue_session_ids:
            if session_id in session_id_mapping:
                new_id = session_id_mapping[session_id]
                connection.execute(
                    sa.text(
                        f"UPDATE {schema}.active_queue_sessions SET session_id = :new_id WHERE session_id = :old_id"
                    ),
                    {"new_id": str(new_id), "old_id": str(session_id)},
                )
        op.add_column(
            "active_queue_sessions",
            sa.Column("sender_name", sa.TEXT(), nullable=True),
            schema=schema,
        )
        op.add_column(
            "active_queue_sessions",
            sa.Column("target_name", sa.TEXT(), nullable=True),
            schema=schema,
        )
        op.add_column(
            "active_queue_sessions",
            sa.Column("task_type", sa.TEXT(), nullable=False),
            schema=schema,
        )
        op.create_unique_constraint(
            "unique_active_queue_session",
            "active_queue_sessions",
            ["session_id", "sender_name", "target_name", "task_type"],
            schema=schema,
        )


def backfill_token_counts(schema: str) -> None:
    """Backfill token counts for existing messages using batch updates."""
    connection = op.get_bind()

    # Initialize tokenizer once outside the loop for performance
    tokenizer = None
    with suppress(Exception):
        tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        if not text:
            return 0
        if tokenizer:
            with suppress(Exception):
                return len(tokenizer.encode(text))
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4

    # Get all messages in batches to handle large datasets
    batch_size = 1000
    offset = 0

    while True:
        result = connection.execute(
            text(
                f"SELECT id, content FROM {schema}.messages LIMIT :limit OFFSET :offset"
            ),
            {"limit": batch_size, "offset": offset},
        )
        messages = result.fetchall()

        if not messages:
            break

        # Calculate token counts and update messages in batches
        batch_updates = []
        for message_id, content in messages:
            token_count = _count_tokens(content)
            batch_updates.append((message_id, token_count))

        # Process updates in smaller batches to stay under statement limits
        update_batch_size = 200
        for i in range(0, len(batch_updates), update_batch_size):
            batch_chunk = batch_updates[i : i + update_batch_size]

            # Use unnest arrays for clean batch update
            ids = [item[0] for item in batch_chunk]
            token_counts = [item[1] for item in batch_chunk]

            connection.execute(
                text(f"""
                    UPDATE {schema}.messages 
                    SET token_count = batch_data.token_count
                    FROM (
                        SELECT UNNEST(:ids) as id, UNNEST(:token_counts) as token_count
                    ) AS batch_data
                    WHERE messages.id = batch_data.id
                """),
                {"ids": ids, "token_counts": token_counts},
            )

        offset += batch_size


def restore_app_user_columns(schema: str, inspector) -> None:
    """Restore app_id and user_id columns to peers and sessions."""
    # Add app_id back to peers
    if not column_exists("peers", "app_id", inspector):
        op.add_column(
            "peers",
            sa.Column("app_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate app_id from workspace_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.peers SET app_id = (
                    SELECT id FROM {schema}.workspaces WHERE name = peers.workspace_name
                )
            """)
        )
        op.alter_column("peers", "app_id", nullable=False, schema=schema)

    # Add app_id back to sessions
    if not column_exists("sessions", "app_id", inspector):
        op.add_column(
            "sessions",
            sa.Column("app_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate app_id from workspace_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.sessions SET app_id = (
                    SELECT id FROM {schema}.workspaces WHERE name = sessions.workspace_name
                )
            """)
        )
        op.alter_column("sessions", "app_id", nullable=False, schema=schema)

    # Add user_id back to sessions
    if not column_exists("sessions", "user_id", inspector):
        op.add_column(
            "sessions",
            sa.Column("user_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate user_id from session_peers
        # The user peer is the one that existed before agent peers were created
        # Agent peers have names that match their IDs (both are nanoids), user peers have different names and IDs
        op.execute(
            sa.text(f"""
                UPDATE {schema}.sessions SET user_id = (
                    SELECT p.id FROM {schema}.peers p
                    JOIN {schema}.session_peers sp ON p.name = sp.peer_name AND p.workspace_name = sp.workspace_name
                    WHERE sp.session_name = sessions.name 
                    AND p.workspace_name = sessions.workspace_name
                    AND p.id != p.name
                    LIMIT 1
                )
            """)
        )

        op.alter_column("sessions", "user_id", nullable=False, schema=schema)


def restore_documents_table(schema: str, inspector) -> None:
    """Restore documents table to pre-peer paradigm state."""
    # Add back id column as primary key
    if not column_exists("documents", "temp_id", inspector):
        op.add_column(
            "documents",
            sa.Column("temp_id", sa.BigInteger(), nullable=True),
            schema=schema,
        )

    op.execute(f"CREATE SEQUENCE IF NOT EXISTS {schema}.documents_id_seq")

    op.execute(f"""
        UPDATE {schema}.documents 
        SET temp_id = nextval('{schema}.documents_id_seq')
        WHERE temp_id IS NULL
    """)

    op.alter_column("documents", "temp_id", nullable=False, schema=schema)

    # Drop current primary key and rename columns
    if constraint_exists("documents", "pk_documents", "primary", inspector):
        op.drop_constraint("pk_documents", "documents", type_="primary", schema=schema)

    op.alter_column("documents", "id", new_column_name="public_id", schema=schema)
    op.alter_column("documents", "temp_id", new_column_name="id", schema=schema)
    op.alter_column(
        "documents", "internal_metadata", new_column_name="metadata", schema=schema
    )

    op.create_primary_key("pk_documents", "documents", ["id"], schema=schema)

    op.alter_column(
        "documents",
        "id",
        nullable=False,
        server_default=sa.text(f"nextval('{schema}.documents_id_seq')"),
        schema=schema,
    )

    op.execute(
        f"ALTER SEQUENCE {schema}.documents_id_seq OWNED BY {schema}.documents.id"
    )

    # Add back user_id and app_id columns
    if not column_exists("documents", "user_id", inspector):
        op.add_column(
            "documents",
            sa.Column("user_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from peer_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.documents SET user_id = (
                    SELECT id FROM {schema}.peers WHERE name = documents.peer_name AND workspace_name = documents.workspace_name
                )
            """)
        )
        op.alter_column("documents", "user_id", nullable=False, schema=schema)

    if not column_exists("documents", "app_id", inspector):
        op.add_column(
            "documents",
            sa.Column("app_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from workspace_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.documents SET app_id = (
                    SELECT id FROM {schema}.workspaces WHERE name = documents.workspace_name
                )
            """)
        )
        op.alter_column("documents", "app_id", nullable=False, schema=schema)

    # Drop new foreign keys
    if fk_exists("documents", "fk_documents_collection_name_collections", inspector):
        op.drop_constraint(
            "fk_documents_collection_name_collections",
            "documents",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("documents", "fk_documents_workspace_name_workspaces", inspector):
        op.drop_constraint(
            "fk_documents_workspace_name_workspaces",
            "documents",
            type_="foreignkey",
            schema=schema,
        )

    op.execute(
        sa.text(f"""
            UPDATE {schema}.documents SET collection_name = (
                SELECT id FROM {schema}.collections WHERE name = documents.collection_name AND workspace_name = documents.workspace_name AND peer_name = documents.peer_name
            )
        """)
    )

    # Restore collection_id from collection_name
    op.alter_column(
        "documents", "collection_name", new_column_name="collection_id", schema=schema
    )

    # Drop new columns
    op.drop_column("documents", "peer_name", schema=schema)
    op.drop_column("documents", "workspace_name", schema=schema)

    # Restore old foreign keys (only fk_ format)
    op.create_foreign_key(
        "fk_documents_collection_id_collections",
        "documents",
        "collections",
        ["collection_id"],
        ["id"],
        referent_schema=schema,
    )

    # Restore old constraint names
    if constraint_exists("documents", "id_length", "check", inspector):
        op.drop_constraint("id_length", "documents", type_="check", schema=schema)
    if constraint_exists("documents", "id_format", "check", inspector):
        op.drop_constraint("id_format", "documents", type_="check", schema=schema)

    op.create_check_constraint(
        "public_id_length", "documents", "length(public_id) = 21", schema=schema
    )
    op.create_check_constraint(
        "public_id_format", "documents", "public_id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def restore_collections_table(schema: str, inspector) -> None:
    """Restore collections table to pre-peer paradigm state."""
    # Add back id column as primary key
    if not column_exists("collections", "temp_id", inspector):
        op.add_column(
            "collections",
            sa.Column("temp_id", sa.BigInteger()),
            schema=schema,
        )

    op.execute(f"CREATE SEQUENCE IF NOT EXISTS {schema}.collections_id_seq")

    op.execute(f"""
        UPDATE {schema}.collections 
        SET temp_id = nextval('{schema}.collections_id_seq')
        WHERE temp_id IS NULL
    """)

    op.alter_column("collections", "temp_id", nullable=False, schema=schema)

    # Drop current primary key and rename columns
    if constraint_exists("collections", "pk_collections", "primary", inspector):
        op.drop_constraint(
            "pk_collections", "collections", type_="primary", schema=schema
        )

    op.alter_column("collections", "id", new_column_name="public_id", schema=schema)
    op.alter_column("collections", "temp_id", new_column_name="id", schema=schema)

    op.create_primary_key("pk_collections", "collections", ["id"], schema=schema)

    op.alter_column(
        "collections",
        "id",
        nullable=False,
        server_default=sa.text(f"nextval('{schema}.collections_id_seq')"),
        schema=schema,
    )

    op.execute(
        f"ALTER SEQUENCE {schema}.collections_id_seq OWNED BY {schema}.collections.id"
    )

    # Add back user_id and app_id columns
    if not column_exists("collections", "user_id", inspector):
        op.add_column(
            "collections",
            sa.Column("user_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from peer_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.collections SET user_id = (
                    SELECT id FROM {schema}.peers WHERE name = collections.peer_name AND workspace_name = collections.workspace_name
                )
            """)
        )
        op.alter_column("collections", "user_id", nullable=False, schema=schema)

    if not column_exists("collections", "app_id", inspector):
        op.add_column(
            "collections",
            sa.Column("app_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from workspace_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.collections SET app_id = (
                    SELECT id FROM {schema}.workspaces WHERE name = collections.workspace_name
                )
            """)
        )
        op.alter_column("collections", "app_id", nullable=False, schema=schema)

    # Drop new foreign keys
    if fk_exists("collections", "fk_collections_peer_name_peers", inspector):
        op.drop_constraint(
            "fk_collections_peer_name_peers",
            "collections",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("collections", "fk_collections_workspace_name_workspaces", inspector):
        op.drop_constraint(
            "fk_collections_workspace_name_workspaces",
            "collections",
            type_="foreignkey",
            schema=schema,
        )

    # Drop new unique constraint
    op.drop_constraint(
        "unique_name_collection_peer", "collections", type_="unique", schema=schema
    )

    # Drop new columns
    op.drop_column("collections", "peer_name", schema=schema)
    op.drop_column("collections", "workspace_name", schema=schema)
    op.drop_column("collections", "internal_metadata", schema=schema)

    # Restore old unique constraint
    op.create_unique_constraint(
        "unique_name_collection_user",
        "collections",
        ["name", "user_id"],
        schema=schema,
    )

    # Restore old constraint names
    if constraint_exists("collections", "id_length", "check", inspector):
        op.drop_constraint("id_length", "collections", type_="check", schema=schema)
    if constraint_exists("collections", "id_format", "check", inspector):
        op.drop_constraint("id_format", "collections", type_="check", schema=schema)

    op.create_check_constraint(
        "public_id_length", "collections", "length(public_id) = 21", schema=schema
    )
    op.create_check_constraint(
        "public_id_format",
        "collections",
        "public_id ~ '^[A-Za-z0-9_-]+$'",
        schema=schema,
    )


def restore_messages_table(schema: str, inspector) -> None:
    """Restore messages table to pre-peer paradigm state."""
    # Add back old columns
    if not column_exists("messages", "session_id", inspector):
        op.add_column(
            "messages",
            sa.Column("session_id", sa.Text(), nullable=True),
            schema=schema,
        )
        op.execute(
            sa.text(f"""
                UPDATE {schema}.messages SET session_id = (
                    SELECT id FROM {schema}.sessions s WHERE s.name = messages.session_name
                )
            """)
        )
        op.alter_column("messages", "session_id", nullable=False, schema=schema)

    if not column_exists("messages", "user_id", inspector):
        op.add_column(
            "messages",
            sa.Column("user_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from peer_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.messages SET user_id = (
                    SELECT id FROM {schema}.peers WHERE name = messages.peer_name AND workspace_name = messages.workspace_name
                )
            """)
        )
        op.alter_column("messages", "user_id", nullable=False, schema=schema)

    if not column_exists("messages", "app_id", inspector):
        op.add_column(
            "messages",
            sa.Column("app_id", sa.TEXT(), nullable=True),
            schema=schema,
        )
        # Populate from workspace_name
        op.execute(
            sa.text(f"""
                UPDATE {schema}.messages SET app_id = (
                    SELECT id FROM {schema}.workspaces WHERE name = messages.workspace_name
                )
            """)
        )
        op.alter_column("messages", "app_id", nullable=False, schema=schema)

    if not column_exists("messages", "is_user", inspector):
        op.add_column(
            "messages",
            sa.Column("is_user", sa.Boolean(), nullable=True),
            schema=schema,
        )
        # Determine is_user based on whether peer_name matches session's user
        op.execute(
            sa.text(f"""
                UPDATE {schema}.messages SET is_user = (
                    SELECT CASE 
                        WHEN s.user_id = messages.peer_name THEN true 
                        ELSE false 
                    END
                    FROM {schema}.sessions s 
                    WHERE s.name = messages.session_name
                )
            """)
        )
        op.alter_column("messages", "is_user", nullable=False, schema=schema)

    # Drop new foreign keys
    if fk_exists("messages", "fk_messages_session_name_sessions", inspector):
        op.drop_constraint(
            "fk_messages_session_name_sessions",
            "messages",
            type_="foreignkey",
            schema=schema,
        )
    if fk_exists("messages", "fk_messages_peer_name_peers", inspector):
        op.drop_constraint(
            "fk_messages_peer_name_peers", "messages", type_="foreignkey", schema=schema
        )
    if fk_exists("messages", "fk_messages_workspace_name_workspaces", inspector):
        op.drop_constraint(
            "fk_messages_workspace_name_workspaces",
            "messages",
            type_="foreignkey",
            schema=schema,
        )

    # Drop new indexes
    if index_exists("messages", "ix_messages_peer_name", inspector):
        op.drop_index("ix_messages_peer_name", table_name="messages", schema=schema)
    if index_exists("messages", "ix_messages_workspace_name", inspector):
        op.drop_index(
            "ix_messages_workspace_name", table_name="messages", schema=schema
        )

    # Drop full text search index
    if index_exists("messages", "idx_messages_content_gin", inspector):
        op.drop_index("idx_messages_content_gin", table_name="messages", schema=schema)

    # Drop new columns
    op.drop_column("messages", "peer_name", schema=schema)
    op.drop_column("messages", "workspace_name", schema=schema)
    op.drop_column("messages", "session_name", schema=schema)
    op.drop_column("messages", "token_count", schema=schema)
    op.drop_column("messages", "internal_metadata", schema=schema)

    # Restore old foreign keys (only fk_ format)
    op.create_foreign_key(
        "fk_messages_session_id_sessions",
        "messages",
        "sessions",
        ["session_id"],
        ["id"],
        referent_schema=schema,
    )


def restore_sessions_table(schema: str, inspector) -> None:
    """Restore sessions table to pre-peer paradigm state."""
    # Add back id column as BigInteger primary key
    if not column_exists("sessions", "temp_id", inspector):
        op.add_column(
            "sessions",
            sa.Column("temp_id", sa.BigInteger()),
            schema=schema,
        )

    op.execute(f"CREATE SEQUENCE IF NOT EXISTS {schema}.sessions_id_seq")

    op.execute(f"""
        UPDATE {schema}.sessions 
        SET temp_id = nextval('{schema}.sessions_id_seq')
        WHERE temp_id IS NULL
    """)

    # Drop current primary key and rename columns
    if constraint_exists("sessions", "pk_sessions", "primary", inspector):
        op.drop_constraint("pk_sessions", "sessions", type_="primary", schema=schema)

    op.alter_column("sessions", "id", new_column_name="public_id", schema=schema)
    op.alter_column("sessions", "temp_id", new_column_name="id", schema=schema)

    op.create_primary_key("pk_sessions", "sessions", ["id"], schema=schema)

    op.alter_column(
        "sessions",
        "id",
        nullable=False,
        server_default=sa.text(f"nextval('{schema}.sessions_id_seq')"),
        schema=schema,
    )

    op.execute(f"ALTER SEQUENCE {schema}.sessions_id_seq OWNED BY {schema}.sessions.id")

    # Drop new foreign keys
    if fk_exists("sessions", "fk_sessions_workspace_name_workspaces", inspector):
        op.drop_constraint(
            "fk_sessions_workspace_name_workspaces",
            "sessions",
            type_="foreignkey",
            schema=schema,
        )

    # Drop new unique constraint
    op.drop_constraint("unique_session_name", "sessions", type_="unique", schema=schema)

    # Drop new columns
    op.drop_column("sessions", "name", schema=schema)
    op.drop_column("sessions", "workspace_name", schema=schema)
    op.drop_column("sessions", "configuration", schema=schema)
    op.drop_column("sessions", "internal_metadata", schema=schema)

    # Restore old constraint names
    if constraint_exists("sessions", "id_length", "check", inspector):
        op.drop_constraint("id_length", "sessions", type_="check", schema=schema)
    if constraint_exists("sessions", "id_format", "check", inspector):
        op.drop_constraint("id_format", "sessions", type_="check", schema=schema)
    if constraint_exists("sessions", "name_length", "check", inspector):
        op.drop_constraint("name_length", "sessions", type_="check", schema=schema)

    op.create_check_constraint(
        "public_id_length", "sessions", "length(public_id) = 21", schema=schema
    )
    op.create_check_constraint(
        "public_id_format", "sessions", "public_id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def restore_peers_table(schema: str, inspector) -> None:
    """Restore peers table to pre-user paradigm state."""
    # Add back id column as BigInteger primary key
    if not column_exists("peers", "temp_id", inspector):
        op.add_column(
            "peers",
            sa.Column("temp_id", sa.BigInteger()),
            schema=schema,
        )

    op.execute(f"CREATE SEQUENCE IF NOT EXISTS {schema}.peers_id_seq")

    op.execute(f"""
        UPDATE {schema}.peers 
        SET temp_id = nextval('{schema}.peers_id_seq')
        WHERE temp_id IS NULL
    """)

    # Drop current primary key and rename columns
    if constraint_exists("peers", "pk_peers", "primary", inspector):
        op.drop_constraint("pk_peers", "peers", type_="primary", schema=schema)

    op.alter_column("peers", "id", new_column_name="public_id", schema=schema)
    op.alter_column("peers", "temp_id", new_column_name="id", schema=schema)

    op.alter_column(
        "peers",
        "id",
        nullable=False,
        server_default=sa.text(f"nextval('{schema}.peers_id_seq')"),
        schema=schema,
    )

    op.execute(f"ALTER SEQUENCE {schema}.peers_id_seq OWNED BY {schema}.peers.id")
    op.create_primary_key("pk_users", "peers", ["id"], schema=schema)

    # Drop new foreign keys
    if fk_exists("peers", "fk_peers_workspace_name_workspaces", inspector):
        op.drop_constraint(
            "fk_peers_workspace_name_workspaces",
            "peers",
            type_="foreignkey",
            schema=schema,
        )

    # Drop new unique constraint and index
    if constraint_exists("peers", "unique_name_workspace_peer", "unique", inspector):
        op.drop_constraint(
            "unique_name_workspace_peer", "peers", type_="unique", schema=schema
        )
    if index_exists("peers", "idx_peers_workspace_lookup", inspector):
        op.drop_index("idx_peers_workspace_lookup", table_name="peers", schema=schema)

    # Restore old unique constraint and index
    op.create_unique_constraint(
        "unique_name_app_user", "peers", ["name", "app_id"], schema=schema
    )
    op.create_index("idx_users_app_lookup", "peers", ["app_id", "name"], schema=schema)

    # Drop new columns
    op.drop_column("peers", "workspace_name", schema=schema)
    op.drop_column("peers", "configuration", schema=schema)
    op.drop_column("peers", "internal_metadata", schema=schema)

    # Restore old constraint names
    if constraint_exists("peers", "id_length", "check", inspector):
        op.drop_constraint("id_length", "peers", type_="check", schema=schema)
    if constraint_exists("peers", "id_format", "check", inspector):
        op.drop_constraint("id_format", "peers", type_="check", schema=schema)

    op.create_check_constraint(
        "public_id_length", "peers", "length(public_id) = 21", schema=schema
    )
    op.create_check_constraint(
        "public_id_format", "peers", "public_id ~ '^[A-Za-z0-9_-]+$'", schema=schema
    )


def restore_workspaces_table(schema: str, inspector) -> None:
    """Restore workspaces table to pre-peer paradigm state (apps)."""
    # Add back id column as BigInteger primary key
    if not column_exists("workspaces", "temp_id", inspector):
        op.add_column(
            "workspaces",
            sa.Column("temp_id", sa.BigInteger()),
            schema=schema,
        )

    op.execute(f"CREATE SEQUENCE IF NOT EXISTS {schema}.workspaces_id_seq")

    op.execute(f"""
        UPDATE {schema}.workspaces 
        SET temp_id = nextval('{schema}.workspaces_id_seq')
        WHERE temp_id IS NULL
    """)

    # Drop current primary key and rename columns
    if constraint_exists("workspaces", "pk_workspaces", "primary", inspector):
        op.drop_constraint(
            "pk_workspaces", "workspaces", type_="primary", schema=schema
        )

    op.alter_column("workspaces", "id", new_column_name="public_id", schema=schema)
    op.alter_column("workspaces", "temp_id", new_column_name="id", schema=schema)

    op.create_primary_key("pk_apps", "workspaces", ["id"], schema=schema)

    op.alter_column(
        "workspaces",
        "id",
        nullable=False,
        server_default=sa.text(f"nextval('{schema}.workspaces_id_seq')"),
        schema=schema,
    )
    op.execute(
        f"ALTER SEQUENCE {schema}.workspaces_id_seq OWNED BY {schema}.workspaces.id"
    )

    # Drop new columns
    op.drop_column("workspaces", "configuration", schema=schema)
    op.drop_column("workspaces", "internal_metadata", schema=schema)

    # Restore old constraint names
    if constraint_exists("workspaces", "id_length", "check", inspector):
        op.drop_constraint("id_length", "workspaces", type_="check", schema=schema)
    if constraint_exists("workspaces", "id_format", "check", inspector):
        op.drop_constraint("id_format", "workspaces", type_="check", schema=schema)

    op.create_check_constraint(
        "public_id_length", "workspaces", "length(public_id) = 21", schema=schema
    )
    op.create_check_constraint(
        "public_id_format",
        "workspaces",
        "public_id ~ '^[A-Za-z0-9_-]+$'",
        schema=schema,
    )


def restore_queue_and_active_queue_sessions_tables(schema: str, inspector) -> None:
    """Restore queue and active_queue_sessions tables to pre-peer paradigm state."""

    connection = op.get_bind()

    # Create reverse mapping from session.public_id (text) back to session.id (BigInteger)
    # At this point in downgrade, sessions table still has both id (BigInteger) and public_id (text)
    session_id_reverse_mapping = {}
    if table_exists("sessions", inspector):
        sessions_mapping = connection.execute(
            sa.text(f"SELECT id, public_id FROM {schema}.sessions")
        ).fetchall()

        for big_int_id, text_id in sessions_mapping:
            session_id_reverse_mapping[text_id] = big_int_id

    # Update queue table
    if table_exists("queue", inspector) and session_id_reverse_mapping:
        # Get current session_id values in queue table (they are text now)
        queue_session_ids = connection.execute(
            sa.text(
                f"SELECT DISTINCT session_id FROM {schema}.queue WHERE session_id IS NOT NULL"
            )
        ).fetchall()

        # Convert session_id values back to BigInteger
        for (session_id,) in queue_session_ids:
            if session_id in session_id_reverse_mapping:
                old_id = session_id_reverse_mapping[session_id]
                connection.execute(
                    sa.text(
                        f"UPDATE {schema}.queue SET session_id = :old_id WHERE session_id = :new_id"
                    ),
                    {"old_id": str(old_id), "new_id": str(session_id)},
                )

        # Change column type back to BigInteger
        op.alter_column(
            "queue",
            "session_id",
            type_=sa.BigInteger(),
            existing_type=sa.Text(),
            postgresql_using="session_id::bigint",
        )

    # Update active_queue_sessions table
    if table_exists("active_queue_sessions", inspector) and session_id_reverse_mapping:
        # Get current session_id values in active_queue_sessions table (they are text now)
        active_queue_session_ids = connection.execute(
            sa.text(
                f"SELECT DISTINCT session_id FROM {schema}.active_queue_sessions WHERE session_id IS NOT NULL"
            )
        ).fetchall()

        if constraint_exists(
            "active_queue_sessions", "pk_active_queue_sessions", "primary", inspector
        ):
            op.drop_constraint(
                "pk_active_queue_sessions",
                "active_queue_sessions",
                type_="primary",
                schema=schema,
            )

        # Convert session_id values back to BigInteger
        for (session_id,) in active_queue_session_ids:
            if session_id in session_id_reverse_mapping:
                old_id = session_id_reverse_mapping[session_id]
                connection.execute(
                    sa.text(
                        f"UPDATE {schema}.active_queue_sessions SET session_id = :old_id WHERE session_id = :new_id"
                    ),
                    {"old_id": str(old_id), "new_id": str(session_id)},
                )

        # Change column type back to BigInteger
        op.alter_column(
            "active_queue_sessions",
            "session_id",
            type_=sa.BigInteger(),
            existing_type=sa.Text(),
            postgresql_using="session_id::bigint",
        )

        op.create_primary_key(
            "pk_active_queue_sessions",
            "active_queue_sessions",
            ["session_id"],
            schema=schema,
        )

        if constraint_exists(
            "active_queue_sessions",
            "unique_active_queue_session",
            "unique",
            inspector,
        ):
            op.drop_constraint(
                "unique_active_queue_session",
                "active_queue_sessions",
                type_="unique",
                schema=schema,
            )

        if column_exists("active_queue_sessions", "id", inspector):
            op.drop_column("active_queue_sessions", "id", schema=schema)
        if column_exists("active_queue_sessions", "sender_name", inspector):
            op.drop_column("active_queue_sessions", "sender_name", schema=schema)
        if column_exists("active_queue_sessions", "target_name", inspector):
            op.drop_column("active_queue_sessions", "target_name", schema=schema)
        if column_exists("active_queue_sessions", "task_type", inspector):
            op.drop_column("active_queue_sessions", "task_type", schema=schema)

    # Restore foreign key constraints
    if table_exists("queue", inspector):
        op.create_foreign_key(
            "fk_queue_session_id_sessions",
            "queue",
            "sessions",
            ["session_id"],
            ["id"],
            referent_schema=schema,
        )

    if table_exists("active_queue_sessions", inspector):
        op.create_foreign_key(
            "fk_active_queue_sessions_session_id_sessions",
            "active_queue_sessions",
            "sessions",
            ["session_id"],
            ["id"],
            referent_schema=schema,
        )


def restore_table_names(schema: str, inspector) -> None:
    """Restore table names: workspaces->apps and peers->users."""
    if inspector.has_table("workspaces", schema=schema):
        op.rename_table("workspaces", "apps", schema=schema)
    if inspector.has_table("peers", schema=schema):
        op.rename_table("peers", "users", schema=schema)


def restore_foreign_keys(schema: str) -> None:
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["public_id"],
        referent_schema=schema,
    )

    op.create_foreign_key(
        "fk_documents_app_id_apps",
        "documents",
        "apps",
        ["app_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_collections_user_id_users",
        "collections",
        "users",
        ["user_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_collections_app_id_apps",
        "collections",
        "apps",
        ["app_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_messages_app_id_apps",
        "messages",
        "apps",
        ["app_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_messages_user_id_users",
        "messages",
        "users",
        ["user_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_sessions_app_id_apps",
        "sessions",
        "apps",
        ["app_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_sessions_user_id_users",
        "sessions",
        "users",
        ["user_id"],
        ["public_id"],
        referent_schema=schema,
    )
    op.create_foreign_key(
        "fk_users_app_id_apps",
        "users",
        "apps",
        ["app_id"],
        ["public_id"],
        referent_schema=schema,
    )
