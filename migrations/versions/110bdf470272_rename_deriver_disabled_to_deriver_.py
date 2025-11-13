"""rename_deriver_disabled_to_deriver_enabled

Revision ID: 110bdf470272
Revises: bb6fb3a7a643
Create Date: 2025-10-31 13:04:31.029856

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "110bdf470272"
down_revision: str | None = "bb6fb3a7a643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """
    Convert deriver_disabled to deriver_enabled in configuration JSONB.

    - deriver_disabled: true  -> deriver_enabled: false
    - deriver_disabled: false -> deriver_enabled: true
    - Remove deriver_disabled key from configuration
    """
    # Update sessions table in batches
    # Combine all cases in one pass: set deriver_enabled based on deriver_disabled value,
    # or just remove deriver_disabled if it's null or other value
    bind = op.get_bind()
    batch_size = 5000

    while True:
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM "{schema}".sessions
                    WHERE configuration ? 'deriver_disabled'
                    ORDER BY id
                    LIMIT :batch_size
                )
                UPDATE "{schema}".sessions s
                SET configuration = CASE
                    WHEN s.configuration->>'deriver_disabled' = 'true' THEN
                        s.configuration - 'deriver_disabled' || jsonb_build_object('deriver_enabled', false)
                    WHEN s.configuration->>'deriver_disabled' = 'false' THEN
                        s.configuration - 'deriver_disabled' || jsonb_build_object('deriver_enabled', true)
                    ELSE
                        s.configuration - 'deriver_disabled'
                END
                FROM batch b
                WHERE s.id = b.id
                  AND s.configuration ? 'deriver_disabled'
                """
            ),
            {"batch_size": batch_size},
        )
        rowcount = result.rowcount
        result.close()
        if rowcount == 0:
            break


def downgrade() -> None:
    """
    Convert deriver_enabled back to deriver_disabled in configuration JSONB.

    - deriver_enabled: false -> deriver_disabled: true
    - deriver_enabled: true  -> deriver_disabled: false
    - Remove deriver_enabled key from configuration
    """
    # Update sessions table in batches
    # Combine all cases in one pass: set deriver_disabled based on deriver_enabled value,
    # or just remove deriver_enabled if it's null or other value
    bind = op.get_bind()
    batch_size = 5000

    while True:
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT id
                    FROM "{schema}".sessions
                    WHERE configuration ? 'deriver_enabled'
                    ORDER BY id
                    LIMIT :batch_size
                )
                UPDATE "{schema}".sessions s
                SET configuration = CASE
                    WHEN s.configuration->>'deriver_enabled' = 'false' THEN
                        s.configuration - 'deriver_enabled' || jsonb_build_object('deriver_disabled', true)
                    WHEN s.configuration->>'deriver_enabled' = 'true' THEN
                        s.configuration - 'deriver_enabled' || jsonb_build_object('deriver_disabled', false)
                    ELSE
                        s.configuration - 'deriver_enabled'
                END
                FROM batch b
                WHERE s.id = b.id
                  AND s.configuration ? 'deriver_enabled'
                """
            ),
            {"batch_size": batch_size},
        )
        rowcount = result.rowcount
        result.close()
        if rowcount == 0:
            break
