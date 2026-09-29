"""add tenants.derivation_paused

Revision ID: 43d77d0c5846
Revises: c4e8a2b91d57
Create Date: 2026-09-21

A per-tenant bit the deriver's claim reads to skip a tenant's work. The
control plane writes it through the tenant registry; the deriver never
decides it, only honours it. Defaults false, so an upgraded deployment
claims exactly as before until something sets the bit.

The partial index covers the claim-side refresh, which reads only the
paused subset: small, and read on a timer by every claiming process.

Both steps are guarded, so a run that dies between them completes on the
next one. ADD COLUMN with a constant default is a catalog-only change on
Postgres 11+, so no table rewrite and no long lock.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "43d77d0c5846"
down_revision: str | None = "c4e8a2b91d57"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

INDEX_NAME = "ix_tenants_derivation_paused"


def upgrade() -> None:
    if not column_exists("tenants", "derivation_paused"):
        op.add_column(
            "tenants",
            sa.Column(
                "derivation_paused",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
            schema=schema,
        )
    if not index_exists("tenants", INDEX_NAME):
        op.create_index(
            INDEX_NAME,
            "tenants",
            ["tenant_id"],
            schema=schema,
            postgresql_where=sa.text("derivation_paused"),
        )


def downgrade() -> None:
    if index_exists("tenants", INDEX_NAME):
        op.drop_index(INDEX_NAME, table_name="tenants", schema=schema)
    if column_exists("tenants", "derivation_paused"):
        op.drop_column("tenants", "derivation_paused", schema=schema)
