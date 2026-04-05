"""codify_document_level_and_times_derived

Revision ID: b8183c5ffb48
Revises: ec8f94139b02
Create Date: 2025-10-31 12:48:54.597269

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from migrations.utils import column_exists, constraint_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "b8183c5ffb48"
down_revision: str | None = "ec8f94139b02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Codify level and times_derived from internal_metadata into explicit columns."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Step 1: Add level column (nullable initially)
    if not column_exists("documents", "level", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "level",
                sa.TEXT(),
                nullable=True,
            ),
            schema=schema,
        )

    # Step 2: Add times_derived column (nullable initially)
    if not column_exists("documents", "times_derived", inspector):
        op.add_column(
            "documents",
            sa.Column("times_derived", sa.Integer(), nullable=True),
            schema=schema,
        )

    # Step 3: Populate level and times_derived from internal_metadata in batches
    # Default to 'explicit' for level and 1 for times_derived if not present in metadata
    batch_size = 5000
    while True:
        result = connection.execute(
            text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM {schema}.documents
                    WHERE level IS NULL OR times_derived IS NULL
                    LIMIT :batch_size
                )
                UPDATE {schema}.documents d
                SET
                    level = COALESCE(
                        d.internal_metadata->>'level',
                        'explicit'
                    ),
                    times_derived = COALESCE(
                        (d.internal_metadata->>'times_derived')::integer,
                        1
                    )
                FROM batch
                WHERE d.id = batch.id
            """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 4: Make level NOT NULL with server default
    op.alter_column(
        "documents",
        "level",
        nullable=False,
        server_default=text("'explicit'"),
        schema=schema,
    )

    # Step 5: Make times_derived NOT NULL with server default
    op.alter_column(
        "documents",
        "times_derived",
        nullable=False,
        server_default=text("1"),
        schema=schema,
    )

    # Step 6: Add CHECK constraint for level
    if not constraint_exists("documents", "level_valid", "check", inspector):
        connection.execute(
            text(
                f"""
                ALTER TABLE {schema}.documents
                ADD CONSTRAINT level_valid
                CHECK (level IN ('explicit', 'deductive'))
                """
            )
        )


def downgrade() -> None:
    """Restore level and times_derived to internal_metadata."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    # Step 1: Drop CHECK constraint for level
    if constraint_exists("documents", "level_valid", "check", inspector):
        connection.execute(
            text(f"ALTER TABLE {schema}.documents DROP CONSTRAINT level_valid")
        )

    # Step 2: Copy level and times_derived back to internal_metadata in batches (optional, for safety)
    batch_size = 5000
    while True:
        result = connection.execute(
            text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM {schema}.documents
                    WHERE internal_metadata IS NULL
                       OR NOT (internal_metadata ? 'level')
                       OR NOT (internal_metadata ? 'times_derived')
                    LIMIT :batch_size
                )
                UPDATE {schema}.documents d
                SET internal_metadata = jsonb_set(
                    jsonb_set(
                        COALESCE(d.internal_metadata, '{{}}'::jsonb),
                        '{{level}}',
                        to_jsonb(d.level)
                    ),
                    '{{times_derived}}',
                    to_jsonb(d.times_derived)
                )
                FROM batch
                WHERE d.id = batch.id
            """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 3: Drop the level column
    if column_exists("documents", "level", inspector):
        op.drop_column("documents", "level", schema=schema)

    # Step 4: Drop the times_derived column
    if column_exists("documents", "times_derived", inspector):
        op.drop_column("documents", "times_derived", schema=schema)
