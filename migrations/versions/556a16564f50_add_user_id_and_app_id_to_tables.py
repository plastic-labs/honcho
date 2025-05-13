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

# revision identifiers, used by Alembic.
revision: str = '556a16564f50'
down_revision: Union[str, None] = 'b765d82110bd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")
    # Add app_id to sessions tabl
    op.add_column("sessions", sa.Column("app_id", sa.Text()), schema=schema)

    # Add app_id and user_id to messages table
    op.add_column("messages", sa.Column("app_id", sa.Text()), schema=schema)
    op.add_column("messages", sa.Column("user_id", sa.Text()), schema=schema)

    # Add app_id to metamessages table
    op.add_column("metamessages", sa.Column("app_id", sa.Text()), schema=schema)

    # Add app_id to collections table
    op.add_column("collections", sa.Column("app_id", sa.Text()), schema=schema)

    # Add app_id and user_id to documents table
    op.add_column("documents", sa.Column("app_id", sa.Text()), schema=schema)
    op.add_column("documents", sa.Column("user_id", sa.Text()), schema=schema)

def downgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    op.drop_column("documents", "user_id", schema=schema)
    op.drop_column("documents", "app_id", schema=schema)

    op.drop_column("metamessages", "app_id", schema=schema)

    op.drop_column("collections", "app_id", schema=schema)

    op.drop_column("messages", "user_id", schema=schema)
    op.drop_column("messages", "app_id", schema=schema)
    
    op.drop_column("sessions", "app_id", schema=schema)