"""add chunk_index to message_embeddings

This migration adds the chunk_index column to message_embeddings table for tracking
chunked message embeddings in external vector stores (turbopuffer/lancedb).

Revision ID: f1a2b3c4d5e6
Revises: baa22cad81e2
Create Date: 2025-11-24 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "baa22cad81e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Add chunk_index column to message_embeddings for tracking chunked message embeddings."""
    inspector = sa.inspect(op.get_bind())

    # Add chunk_index column to message_embeddings if it doesn't exist
    # This is needed to track which chunk of a message this embedding represents
    # Vector ID format: {message_public_id}_{chunk_index}
    if not column_exists("message_embeddings", "chunk_index", inspector):
        op.add_column(
            "message_embeddings",
            sa.Column(
                "chunk_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )


def downgrade() -> None:
    """Remove chunk_index column from message_embeddings."""
    inspector = sa.inspect(op.get_bind())

    # Remove chunk_index column if it exists
    if column_exists("message_embeddings", "chunk_index", inspector):
        op.drop_column("message_embeddings", "chunk_index", schema=schema)
