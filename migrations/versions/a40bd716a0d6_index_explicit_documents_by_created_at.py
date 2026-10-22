"""index explicit documents by created_at for the dream-due poll

The dream-due poll (src/dreamer/dream_due.py) finds collections whose newest
explicit document falls inside a lookback window (DREAM.DUE_LOOKBACK_HOURS),
instead of aggregating every explicit document of every collection. This partial
index serves that range: one short scan per partition over the window's
explicit documents. It doesn't lead with tenant_id because the poll runs
across tenants on the service session.

The build holds no long write lock (see migrations.utils.create_index_online).

Revision ID: a40bd716a0d6
Revises: 647b773f05ac
Create Date: 2026-10-22

"""

from collections.abc import Sequence

from migrations.utils import create_index_online, drop_index_online

# revision identifiers, used by Alembic.
revision: str = "a40bd716a0d6"
down_revision: str | None = "647b773f05ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The same as in src/models.py.
NAME = "ix_documents_explicit_created_at"
TABLE = "documents"


def upgrade() -> None:
    create_index_online(NAME, TABLE, "created_at", where="level = 'explicit'")


def downgrade() -> None:
    drop_index_online(NAME, TABLE)
