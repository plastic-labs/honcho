"""make tenants.vector_correlation_id unique

Revision ID: 60fc2270fab7
Revises: 29093eb3eeda
Create Date: 2026-10-12

A tenant's vector_correlation_id names its vector-store namespaces, so two
tenants holding the same key would share namespaces: deleting one tenant's
workspace would delete the other's vectors. Nothing enforced that the key is
unique; this replaces the plain index from e5fe7f8bcf62 with a unique one over
the non-NULL keys. Nothing looks tenants up by the key, so the plain index has
no reader of its own and the unique one replaces it rather than joining it.

Before building the index, the upgrade looks for keys that are already
duplicated and refuses with every one of them named, instead of failing on a
bare unique violation that reports only the first. Every step is guarded, so
a run that dies part-way completes on the next one.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

from migrations.utils import get_schema, index_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = "60fc2270fab7"
down_revision: str | None = "29093eb3eeda"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

UNIQUE_INDEX = "uq_tenants_vector_correlation_id"
PLAIN_INDEX = "ix_tenants_vector_correlation_id"


def raise_on_duplicate_keys(conn: Connection, schema_name: str) -> None:
    """Refuse the upgrade if any vector_correlation_id is held by more than
    one tenant, naming every duplicated key and the tenants holding it."""
    duplicates = conn.execute(
        sa.text(
            "SELECT vector_correlation_id, array_agg(tenant_id ORDER BY tenant_id) "
            + f'FROM "{schema_name}"."tenants" '
            + "WHERE vector_correlation_id IS NOT NULL "
            + "GROUP BY vector_correlation_id HAVING count(*) > 1 "
            + "ORDER BY vector_correlation_id"
        )
    ).all()
    if duplicates:
        listing = "; ".join(f"{key!r}: {list(tenants)}" for key, tenants in duplicates)
        raise RuntimeError(
            "Cannot make tenants.vector_correlation_id unique: these keys are "
            + f"held by more than one tenant ({listing}). Each key names a "
            + "vector-store namespace, so resolve every duplicate before upgrading."
        )


def upgrade() -> None:
    if not table_exists("tenants"):
        return
    if not index_exists("tenants", UNIQUE_INDEX):
        raise_on_duplicate_keys(op.get_bind(), schema)
        op.create_index(
            UNIQUE_INDEX,
            "tenants",
            ["vector_correlation_id"],
            unique=True,
            schema=schema,
            postgresql_where=sa.text("vector_correlation_id IS NOT NULL"),
        )
    if index_exists("tenants", PLAIN_INDEX):
        op.drop_index(PLAIN_INDEX, table_name="tenants", schema=schema)


def downgrade() -> None:
    if not table_exists("tenants"):
        return
    if not index_exists("tenants", PLAIN_INDEX):
        op.create_index(
            PLAIN_INDEX, "tenants", ["vector_correlation_id"], schema=schema
        )
    if index_exists("tenants", UNIQUE_INDEX):
        op.drop_index(UNIQUE_INDEX, table_name="tenants", schema=schema)
