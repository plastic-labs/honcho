"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from migrations.verification import (
    MigrationVerifier,
    run_downgrade_checks,
    run_upgrade_checks,
)
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}
    run_upgrade_checks(revision, verify_upgrade)


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
    run_downgrade_checks(revision, verify_downgrade)


def verify_upgrade(verifier: MigrationVerifier) -> None:
    """Add post-upgrade assertions that validate the schema/data."""
    pass


def verify_downgrade(verifier: MigrationVerifier) -> None:
    """Add post-downgrade assertions that validate the rollback."""
    pass
