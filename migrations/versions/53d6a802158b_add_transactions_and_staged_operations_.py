"""add_transactions_and_staged_operations_tables

Revision ID: 53d6a802158b
Revises: b765d82110bd
Create Date: 2025-05-13 10:54:17.841226

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '53d6a802158b'
down_revision: Union[str, None] = 'b765d82110bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create transactions table
    op.create_table(
        'transactions',
        sa.Column('transaction_id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.PrimaryKeyConstraint('transaction_id'),
        sa.CheckConstraint(
            "status IN ('pending', 'committed', 'failed', 'rolled_back', 'expired')",
            name='valid_status_check'
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name='valid_expiration_check'
        )
    )
    
    # Create staged_operations table
    op.create_table(
        'staged_operations',
        sa.Column('operation_id', sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column('transaction_id', sa.BigInteger(), nullable=False),
        sa.Column('sequence_number', sa.BigInteger(), nullable=False),
        sa.Column('parameters', JSONB(), nullable=False),
        sa.Column('payload', JSONB(), nullable=False),
        sa.Column('handler_function', sa.Text(), nullable=False),
        sa.Column('resource_public_id', sa.Text(), nullable=True),
        sa.Column('schema_arg_name', sa.Text(), nullable=True),
        sa.Column('is_list_schema', sa.Boolean(), server_default='false', nullable=False),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.transaction_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('operation_id'),
        sa.UniqueConstraint('transaction_id', 'sequence_number', name='uq_transaction_sequence')
    )
    op.create_index(
        'idx_staged_operations_transaction_sequence',
        'staged_operations',
        ['transaction_id', 'sequence_number'],
        unique=False
    )


def downgrade() -> None:
    op.drop_table('staged_operations')
    op.drop_table('transactions')
