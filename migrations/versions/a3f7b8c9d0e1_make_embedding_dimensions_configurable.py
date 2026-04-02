"""make embedding dimensions configurable

This is a no-op migration that documents the change from hardcoded 1536-dimension
vectors to configurable dimensions via VECTOR_STORE_DIMENSIONS setting.

Existing deployments using the default 1536 dimensions are unaffected.
Changing VECTOR_STORE_DIMENSIONS to a different value requires either:
  - A fresh database, or
  - Manually altering the embedding columns and re-indexing all embeddings.

The Vector column type in SQLAlchemy models now reads from settings.VECTOR_STORE.DIMENSIONS
at class definition time. This migration exists to mark the schema's logical dependency
on that configuration value.

Revision ID: a3f7b8c9d0e1
Revises: 119a52b73c60
Create Date: 2025-12-01 00:00:00.000000

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "a3f7b8c9d0e1"
down_revision: str | None = "119a52b73c60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No-op: embedding dimensions are now read from VECTOR_STORE_DIMENSIONS setting.
    # The default (1536) matches the previous hardcoded value.
    pass


def downgrade() -> None:
    # No-op: reverting to hardcoded 1536 is handled by reverting the code change.
    pass
