"""add_session_name_column_to_documents

Revision ID: 564ba40505c5
Revises: 88b0fb10906f
Create Date: 2025-10-01 15:32:13.210971

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, fk_exists, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "564ba40505c5"
down_revision: str | None = "88b0fb10906f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add session_name column to documents table and migrate data from internal_metadata."""
    schema = settings.DB.SCHEMA
    inspector = sa.inspect(op.get_bind())

    # Step 1: Add session_name column as nullable
    if not column_exists("documents", "session_name", inspector):
        op.add_column(
            "documents",
            sa.Column("session_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Step 2: Migrate data from internal_metadata to session_name column in batches
    # Process in batches to avoid timeout with large datasets
    bind = op.get_bind()
    batch_size = 5000

    while True:
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM {schema}.documents
                    WHERE internal_metadata ? 'session_name'
                    AND session_name IS NULL
                    LIMIT :batch_size
                )
                UPDATE {schema}.documents
                SET session_name = internal_metadata->>'session_name'
                FROM batch
                WHERE documents.id = batch.id
                """
            ),
            {"batch_size": batch_size},
        )

        if result.rowcount == 0:
            break

    # Step 3: Create index on session_name for efficient querying
    if not index_exists("documents", "idx_documents_session_name", inspector):
        op.create_index(
            "idx_documents_session_name",
            "documents",
            ["session_name"],
            schema=schema,
        )

    # Step 4: Add foreign key constraint to sessions table
    # Documents can have NULL session_name (for global observations not tied to a session)
    if not fk_exists("documents", "fk_documents_session_workspace", inspector):
        op.create_foreign_key(
            "fk_documents_session_workspace",
            "documents",
            "sessions",
            ["session_name", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Note: We keep session_name nullable since some documents may not have a session


def downgrade() -> None:
    """Remove session_name column and restore data to internal_metadata."""
    schema = settings.DB.SCHEMA
    inspector = sa.inspect(op.get_bind())

    # Step 1: Copy session_name back to internal_metadata
    op.execute(
        sa.text(
            f"""
            UPDATE {schema}.documents
            SET internal_metadata = internal_metadata || jsonb_build_object('session_name', session_name)
            WHERE session_name IS NOT NULL
            """
        )
    )

    # Step 2: Drop foreign key constraint
    if fk_exists("documents", "fk_documents_session_workspace", inspector):
        op.drop_constraint(
            "fk_documents_session_workspace",
            "documents",
            type_="foreignkey",
            schema=schema,
        )

    # Step 3: Drop index
    if index_exists("documents", "idx_documents_session_name", inspector):
        op.drop_index("idx_documents_session_name", "documents", schema=schema)

    # Step 4: Drop session_name column
    if column_exists("documents", "session_name", inspector):
        op.drop_column("documents", "session_name", schema=schema)
