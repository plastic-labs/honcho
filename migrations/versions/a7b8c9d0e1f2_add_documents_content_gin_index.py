"""add_documents_content_gin_index

Add GIN index on to_tsvector('english', content) for the documents table.
This enables efficient full-text search over document content during hybrid retrieval,
avoiding sequential scans on every FTS query.

Revision ID: a7b8c9d0e1f2
Revises: f1a2b3c4d5e6
Create Date: 2026-05-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not index_exists("documents", "idx_documents_content_gin", inspector):
        op.create_index(
            "idx_documents_content_gin",
            "documents",
            [sa.text("to_tsvector('english', content)")],
            postgresql_using="gin",
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if index_exists("documents", "idx_documents_content_gin", inspector):
        op.drop_index(
            "idx_documents_content_gin",
            table_name="documents",
            schema=schema,
        )
