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
    # Update sessions table
    # Set deriver_enabled to false where deriver_disabled is true
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_disabled' || jsonb_build_object('deriver_enabled', false)
            WHERE configuration ? 'deriver_disabled'
              AND (configuration->>'deriver_disabled')::boolean = true
            """
        )
    )

    # Set deriver_enabled to true where deriver_disabled is false
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_disabled' || jsonb_build_object('deriver_enabled', true)
            WHERE configuration ? 'deriver_disabled'
              AND (configuration->>'deriver_disabled')::boolean = false
            """
        )
    )

    # Remove any remaining deriver_disabled keys (for null or other values)
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_disabled'
            WHERE configuration ? 'deriver_disabled'
            """
        )
    )


def downgrade() -> None:
    """
    Convert deriver_enabled back to deriver_disabled in configuration JSONB.

    - deriver_enabled: false -> deriver_disabled: true
    - deriver_enabled: true  -> deriver_disabled: false
    - Remove deriver_enabled key from configuration
    """
    # Update sessions table
    # Set deriver_disabled to true where deriver_enabled is false
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_enabled' || jsonb_build_object('deriver_disabled', true)
            WHERE configuration ? 'deriver_enabled'
              AND (configuration->>'deriver_enabled')::boolean = false
            """
        )
    )

    # Set deriver_disabled to false where deriver_enabled is true
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_enabled' || jsonb_build_object('deriver_disabled', false)
            WHERE configuration ? 'deriver_enabled'
              AND (configuration->>'deriver_enabled')::boolean = true
            """
        )
    )

    # Remove any remaining deriver_enabled keys (for null or other values)
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}".sessions
            SET configuration = configuration - 'deriver_enabled'
            WHERE configuration ? 'deriver_enabled'
            """
        )
    )
