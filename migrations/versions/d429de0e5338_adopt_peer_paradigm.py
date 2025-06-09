"""adopt peer paradigm

Revision ID: d429de0e5338
Revises: 66e63cf2cf77
Create Date: 2025-06-09 15:16:38.164067

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from nanoid import generate as generate_nanoid
from sqlalchemy.dialects import postgresql

from migrations.utils import column_exists, fk_exists, index_exists

# revision identifiers, used by Alembic.
revision: str = 'd429de0e5338'
down_revision: Union[str, None] = '66e63cf2cf77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    # Rename tables if they exist
    if inspector.has_table("apps"):
        op.rename_table('apps', 'workspaces')
    if inspector.has_table("users"):
        op.rename_table('users', 'peers')

    # Alter columns in renamed tables
    if column_exists('peers', 'app_id', inspector):
        op.alter_column('peers', 'app_id', new_column_name='workspace_id')

    # Add workspace_id to sessions table
    if not column_exists('sessions', 'workspace_id', inspector):
        op.add_column('sessions', sa.Column('workspace_id', sa.TEXT(), nullable=True))
    
    # Data Migration: Populate sessions.workspace_id
    if column_exists('sessions', 'app_id', inspector):
        op.execute("UPDATE sessions SET workspace_id = app_id")

    op.alter_column('sessions', 'workspace_id', existing_type=sa.TEXT(), nullable=False)

    # Create session_peers table for many-to-many relationship
    if not inspector.has_table('session_peers'):
        op.create_table('session_peers',
            sa.Column('session_public_id', sa.TEXT(), nullable=False),
            sa.Column('peer_public_id', sa.TEXT(), nullable=False),
            sa.ForeignKeyConstraint(['peer_public_id'], ['peers.public_id'], ),
            sa.ForeignKeyConstraint(['session_public_id'], ['sessions.public_id'], ),
            sa.PrimaryKeyConstraint('session_public_id', 'peer_public_id')
        )

    # --- Data Migration ---
    conn = op.get_bind()

    # Create an 'agent' peer for each workspace
    workspaces = conn.execute(sa.text("SELECT public_id FROM workspaces")).fetchall()
    agent_peers_map = {}
    for workspace_data in workspaces:
        workspace_id = workspace_data[0]
        agent_peer_public_id = generate_nanoid()
        agent_peers_map[workspace_id] = agent_peer_public_id
        op.execute(
            sa.text(
                "INSERT INTO peers (public_id, name, workspace_id, created_at, metadata) VALUES (:public_id, 'agent', :workspace_id, NOW(), '{}'::jsonb)"
            ).bindparams(public_id=agent_peer_public_id, workspace_id=workspace_id)
        )

    # Fetch existing sessions to migrate relationships
    if column_exists('sessions', 'user_id', inspector) and column_exists('sessions', 'workspace_id', inspector):
        sessions = conn.execute(sa.text("SELECT public_id, user_id, workspace_id FROM sessions")).fetchall()

        # Populate session_peers with both the original user and the new agent
        for session_data in sessions:
            session_id, user_id, workspace_id = session_data
            if user_id:
                op.execute(sa.text("INSERT INTO session_peers (session_public_id, peer_public_id) VALUES (:sid, :pid)").bindparams(sid=session_id, pid=user_id))
            agent_peer_id = agent_peers_map.get(workspace_id)
            if agent_peer_id:
                op.execute(sa.text("INSERT INTO session_peers (session_public_id, peer_public_id) VALUES (:sid, :pid)").bindparams(sid=session_id, pid=agent_peer_id))

    # Add and populate the new sender_id column in messages
    if not column_exists('messages', 'sender_id', inspector):
        op.add_column('messages', sa.Column('sender_id', sa.TEXT(), nullable=True))

    if column_exists('messages', 'is_user', inspector):
        # Set sender_id for user messages
        op.execute("""
            UPDATE messages
            SET sender_id = s.user_id
            FROM sessions s
            WHERE messages.session_id = s.public_id AND messages.is_user = TRUE
        """)
        # Set sender_id for agent messages
        op.execute("""
            UPDATE messages m
            SET sender_id = p.public_id
            FROM sessions s
            JOIN peers p ON s.workspace_id = p.workspace_id
            WHERE m.session_id = s.public_id
              AND p.name = 'agent'
              AND m.is_user = FALSE
        """)

    # Finalize schema: add constraints, drop old columns
    op.alter_column('messages', 'sender_id', existing_type=sa.TEXT(), nullable=False)
    op.alter_column('messages', 'session_id', existing_type=sa.TEXT(), nullable=True)

    if not fk_exists('messages', 'messages_sender_id_fkey', inspector):
        op.create_foreign_key('messages_sender_id_fkey', 'messages', 'peers', ['sender_id'], ['public_id'])

    if column_exists('messages', 'is_user', inspector):
        op.drop_column('messages', 'is_user')
    
    if index_exists('messages', 'idx_messages_session_lookup', inspector):
        op.drop_index('idx_messages_session_lookup', table_name='messages')
    op.create_index('idx_messages_session_lookup', 'messages', ['session_id', 'id'], unique=False, postgresql_include=['public_id', 'created_at'])

    # Clean up sessions table
    if fk_exists('sessions', 'sessions_user_id_fkey', inspector):
        op.drop_constraint('sessions_user_id_fkey', 'sessions', type_='foreignkey')
    if fk_exists('sessions', 'sessions_app_id_fkey', inspector):
        op.drop_constraint('sessions_app_id_fkey', 'sessions', type_='foreignkey')
    if column_exists('sessions', 'user_id', inspector):
        op.drop_column('sessions', 'user_id')
    if column_exists('sessions', 'app_id', inspector):
        op.drop_column('sessions', 'app_id')

    if not fk_exists('sessions', 'sessions_workspace_id_fkey', inspector):
        op.create_foreign_key('sessions_workspace_id_fkey', 'sessions', 'workspaces', ['workspace_id'], ['public_id'])

    # Drop metamessages table
    if inspector.has_table('metamessages'):
        op.drop_table('metamessages')

    # Update other tables
    if column_exists('collections', 'user_id', inspector):
        op.alter_column('collections', 'user_id', new_column_name='peer_id')
    if column_exists('collections', 'app_id', inspector):
        op.alter_column('collections', 'app_id', new_column_name='workspace_id')

    if column_exists('documents', 'user_id', inspector):
        op.alter_column('documents', 'user_id', new_column_name='peer_id')
    if column_exists('documents', 'app_id', inspector):
        op.alter_column('documents', 'app_id', new_column_name='workspace_id')


def downgrade() -> None:
    # Note: This downgrade path only reverts schema changes and does not restore data.
    inspector = sa.inspect(op.get_bind())

    if column_exists('documents', 'peer_id', inspector):
        op.alter_column('documents', 'peer_id', new_column_name='user_id')
    if column_exists('documents', 'workspace_id', inspector):
        op.alter_column('documents', 'workspace_id', new_column_name='app_id')

    if column_exists('collections', 'peer_id', inspector):
        op.alter_column('collections', 'peer_id', new_column_name='user_id')
    if column_exists('collections', 'workspace_id', inspector):
        op.alter_column('collections', 'workspace_id', new_column_name='app_id')
    
    if not inspector.has_table('metamessages'):
        op.create_table('metamessages',
            sa.Column('id', sa.BIGINT(), autoincrement=True, nullable=False),
            sa.Column('public_id', sa.TEXT(), autoincrement=False, nullable=False),
            sa.Column('session_id', sa.TEXT(), autoincrement=False, nullable=False),
            sa.Column('content', sa.TEXT(), autoincrement=False, nullable=False),
            sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False),
            sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False),
            sa.ForeignKeyConstraint(['session_id'], ['sessions.public_id'], name='metamessages_session_id_fkey'),
            sa.PrimaryKeyConstraint('id', name='metamessages_pkey')
        )
    
    if fk_exists('sessions', 'sessions_workspace_id_fkey', inspector):
        op.drop_constraint('sessions_workspace_id_fkey', 'sessions', type_='foreignkey')

    if not column_exists('sessions', 'app_id', inspector):
        op.add_column('sessions', sa.Column('app_id', sa.TEXT(), autoincrement=False, nullable=True))
    if column_exists('sessions', 'workspace_id', inspector):
        op.execute("UPDATE sessions SET app_id = workspace_id")
        op.drop_column('sessions', 'workspace_id')

    if not column_exists('sessions', 'user_id', inspector):
        op.add_column('sessions', sa.Column('user_id', sa.TEXT(), autoincrement=False, nullable=True))
    
    if not column_exists('messages', 'is_user', inspector):
        op.add_column('messages', sa.Column('is_user', sa.BOOLEAN(), autoincrement=False, nullable=True))
    
    if fk_exists('messages', 'messages_sender_id_fkey', inspector):
        op.drop_constraint('messages_sender_id_fkey', 'messages', type_='foreignkey')

    op.alter_column('messages', 'session_id', existing_type=sa.TEXT(), nullable=False)
    if column_exists('messages', 'sender_id', inspector):
        op.drop_column('messages', 'sender_id')

    if inspector.has_table('session_peers'):
        op.drop_table('session_peers')
    
    if inspector.has_table("peers"):
        op.rename_table('peers', 'users')
    if inspector.has_table("workspaces"):
        op.rename_table('workspaces', 'apps')
    
    if column_exists('users', 'workspace_id', inspector):
        op.alter_column('users', 'workspace_id', new_column_name='app_id')
