"""Add ActiveQueueUser table for user-level concurrency control

Revision ID: e6b49985cb0c
Revises: 66e63cf2cf77
Create Date: 2025-06-11 18:22:22.704169

"""
from typing import Sequence, Union
from os import getenv

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6b49985cb0c'
down_revision: Union[str, None] = '66e63cf2cf77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
schema = getenv("DATABASE_SCHEMA", "public")


def upgrade() -> None:
    # Create the active_queue_users table
    op.create_table('active_queue_users',
        sa.Column('user_id', sa.TEXT(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('user_id'),
        schema=schema
    )
    
    # Create index on user_id for faster lookups
    op.create_index('ix_active_queue_users_user_id', 'active_queue_users', ['user_id'], schema=schema)


def downgrade() -> None:
    # Drop the index first
    op.drop_index('ix_active_queue_users_user_id', table_name='active_queue_users', schema=schema)
    
    # Drop the table
    op.drop_table('active_queue_users', schema=schema)
