"""add_documents_collection_level_index

Add composite index on (workspace_name, observer, observed, level) for the
documents table. Speeds up the explicit-document COUNT queries issued by the
dream scheduler and orchestrator guard-pair write.

Revision ID: b2c3d4e5f6a7
Revises: a7b8c9d0e1f2
Create Date: 2026-05-07

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not index_exists("documents", "ix_documents_collection_level", inspector):
        op.create_index(
            "ix_documents_collection_level",
            "documents",
            ["workspace_name", "observer", "observed", "level"],
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if index_exists("documents", "ix_documents_collection_level", inspector):
        op.drop_index(
            "ix_documents_collection_level",
            table_name="documents",
            schema=schema,
        )
