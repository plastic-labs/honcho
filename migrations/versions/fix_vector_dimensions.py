"""Update vector columns to use configurable dimensions.

Revision ID: a1b2c3d4e5f7
Revises: e4eba9cfaa6f
Create Date: 2026-04-01
"""

from alembic import op

revision = "a1b2c3d4e5f7"
down_revision = "e4eba9cfaa6f"
branch_labels = None
depends_on = None


def get_configured_dimensions():
    import os
    return int(os.environ.get("VECTOR_STORE_DIMENSIONS", "1024"))


def upgrade() -> None:
    dims = get_configured_dimensions()
    # Alter vector columns to match configured dimensions
    op.execute(f"ALTER TABLE message_embeddings ALTER COLUMN embedding TYPE vector({dims})")
    op.execute(f"ALTER TABLE documents ALTER COLUMN embedding TYPE vector({dims})")


def downgrade() -> None:
    op.execute("ALTER TABLE message_embeddings ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1536)")
