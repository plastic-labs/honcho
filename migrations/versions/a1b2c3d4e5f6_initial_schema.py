"""Initial schema creation

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create apps table
    op.create_table('apps',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(name) <= 512", name='name_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_apps')),
        sa.UniqueConstraint('name', name=op.f('uq_apps_name')),
        sa.UniqueConstraint('public_id', name=op.f('uq_apps_public_id'))
    )
    op.create_index(op.f('ix_apps_created_at'), 'apps', ['created_at'], unique=False)
    op.create_index(op.f('ix_apps_id'), 'apps', ['id'], unique=False)
    op.create_index(op.f('ix_apps_name'), 'apps', ['name'], unique=False)
    op.create_index(op.f('ix_apps_public_id'), 'apps', ['public_id'], unique=False)

    # Create users table
    op.create_table('users',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('app_id', sa.Text(), nullable=False),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(name) <= 512", name='name_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['app_id'], ['apps.public_id'], name=op.f('fk_users_app_id_apps')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('name', 'app_id', name='unique_name_app_user'),
        sa.UniqueConstraint('public_id', name=op.f('uq_users_public_id'))
    )
    op.create_index(op.f('ix_users_app_id'), 'users', ['app_id'], unique=False)
    op.create_index(op.f('ix_users_created_at'), 'users', ['created_at'], unique=False)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_name'), 'users', ['name'], unique=False)
    op.create_index(op.f('ix_users_public_id'), 'users', ['public_id'], unique=False)

    # Create sessions table
    op.create_table('sessions',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['user_id'], ['users.public_id'], name=op.f('fk_sessions_user_id_users')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_sessions')),
        sa.UniqueConstraint('public_id', name=op.f('uq_sessions_public_id'))
    )
    op.create_index(op.f('ix_sessions_created_at'), 'sessions', ['created_at'], unique=False)
    op.create_index(op.f('ix_sessions_id'), 'sessions', ['id'], unique=False)
    op.create_index(op.f('ix_sessions_public_id'), 'sessions', ['public_id'], unique=False)
    op.create_index(op.f('ix_sessions_user_id'), 'sessions', ['user_id'], unique=False)

    # Create messages table
    op.create_table('messages',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('session_id', sa.Text(), nullable=False),
        sa.Column('is_user', sa.Boolean(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(content) <= 65535", name='content_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.public_id'], name=op.f('fk_messages_session_id_sessions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_messages')),
        sa.UniqueConstraint('public_id', name=op.f('uq_messages_public_id'))
    )
    op.create_index(op.f('ix_messages_created_at'), 'messages', ['created_at'], unique=False)
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    op.create_index(op.f('ix_messages_public_id'), 'messages', ['public_id'], unique=False)
    op.create_index(op.f('ix_messages_session_id'), 'messages', ['session_id'], unique=False)

    # Create metamessages table
    op.create_table('metamessages',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('metamessage_type', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('message_id', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(content) <= 65535", name='content_length'),
        sa.CheckConstraint("length(metamessage_type) <= 512", name='metamessage_type_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['message_id'], ['messages.public_id'], name=op.f('fk_metamessages_message_id_messages')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_metamessages')),
        sa.UniqueConstraint('public_id', name=op.f('uq_metamessages_public_id'))
    )
    op.create_index(op.f('ix_metamessages_created_at'), 'metamessages', ['created_at'], unique=False)
    op.create_index(op.f('ix_metamessages_id'), 'metamessages', ['id'], unique=False)
    op.create_index(op.f('ix_metamessages_message_id'), 'metamessages', ['message_id'], unique=False)
    op.create_index(op.f('ix_metamessages_metamessage_type'), 'metamessages', ['metamessage_type'], unique=False)
    op.create_index(op.f('ix_metamessages_public_id'), 'metamessages', ['public_id'], unique=False)

    # Create collections table
    op.create_table('collections',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(name) <= 512", name='name_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['user_id'], ['users.public_id'], name=op.f('fk_collections_user_id_users')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_collections')),
        sa.UniqueConstraint('name', 'user_id', name='unique_name_collection_user'),
        sa.UniqueConstraint('public_id', name=op.f('uq_collections_public_id'))
    )
    op.create_index(op.f('ix_collections_created_at'), 'collections', ['created_at'], unique=False)
    op.create_index(op.f('ix_collections_id'), 'collections', ['id'], unique=False)
    op.create_index(op.f('ix_collections_name'), 'collections', ['name'], unique=False)
    op.create_index(op.f('ix_collections_public_id'), 'collections', ['public_id'], unique=False)
    op.create_index(op.f('ix_collections_user_id'), 'collections', ['user_id'], unique=False)

    # Create documents table
    op.create_table('documents',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('public_id', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('collection_id', sa.Text(), nullable=False),
        sa.CheckConstraint("length(public_id) = 21", name='public_id_length'),
        sa.CheckConstraint("length(content) <= 65535", name='content_length'),
        sa.CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name='public_id_format'),
        sa.ForeignKeyConstraint(['collection_id'], ['collections.public_id'], name=op.f('fk_documents_collection_id_collections')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_documents')),
        sa.UniqueConstraint('public_id', name=op.f('uq_documents_public_id'))
    )
    op.create_index(op.f('ix_documents_collection_id'), 'documents', ['collection_id'], unique=False)
    op.create_index(op.f('ix_documents_created_at'), 'documents', ['created_at'], unique=False)
    op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
    op.create_index(op.f('ix_documents_public_id'), 'documents', ['public_id'], unique=False)

    # Create queue table
    op.create_table('queue',
        sa.Column('id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('session_id', sa.BigInteger(), nullable=False),
        sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], name=op.f('fk_queue_session_id_sessions')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_queue'))
    )
    op.create_index(op.f('ix_queue_session_id'), 'queue', ['session_id'], unique=False)

    # Create active_queue_sessions table
    op.create_table('active_queue_sessions',
        sa.Column('session_id', sa.BigInteger(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], name=op.f('fk_active_queue_sessions_session_id_sessions')),
        sa.PrimaryKeyConstraint('session_id', name=op.f('pk_active_queue_sessions'))
    )
    op.create_index(op.f('ix_active_queue_sessions_session_id'), 'active_queue_sessions', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_table('active_queue_sessions')
    op.drop_table('queue')
    op.drop_table('documents')
    op.drop_table('collections')
    op.drop_table('metamessages')
    op.drop_table('messages')
    op.drop_table('sessions')
    op.drop_table('users')
    op.drop_table('apps') 