"""add tenant_id first class primitive and shared partitioned schema

Bootstraps the shared-tenant target schema: a ``tenants`` registry table, a
``tenant_id`` column on every tenant-scoped table, composite ``(tenant_id, id)``
primary keys, composite foreign keys, and ``HASH(tenant_id)`` partitioning on the
data tables. The service tables (``queue``, ``active_queue_sessions``) carry a
plain ``tenant_id`` and stay unpartitioned.

This CREATES the partitioned tables from scratch — partitioning cannot be
introduced by ``ALTER TABLE`` — so it bootstraps a fresh shared database that
per-tenant data is consolidated into, rather than transforming an existing
single-tenant database in place.

The DDL is generated from the declarative models (``src.models.Base.metadata``)
rather than hand-transcribed. The models are the single source of truth for the
target schema; the compiler renders ``PARTITION BY`` from each table's
``postgresql_partition_by`` option and applies the naming convention; and this
keeps the bootstrap exactly in sync with the accepted data model. Only the
per-table HASH partitions (which the model layer does not enumerate) are created
explicitly. The target schema and ``search_path`` are established by
``migrations/env.py`` before this runs.

Revision ID: 09dfcd158c6c
Revises: e4eba9cfaa6f
Create Date: 2026-09-02 11:38:27.736205

"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import Table
from sqlalchemy.schema import CreateIndex, CreateTable

# Importing Base from src.models (rather than src.db) also registers every model
# on Base.metadata as a side effect — that populated metadata is the whole schema
# this migration builds.
from src.models import Base  # pyright: ignore[reportPrivateLocalImportUsage]

# revision identifiers, used by Alembic.
revision: str = "09dfcd158c6c"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# HASH(tenant_id) partition count for the tenant-scoped tables. Sized so the
# largest table (a couple hundred GB fleet-wide) lands at roughly a couple GB per
# partition — small enough to stay maintainable and well under a node's memory,
# with headroom for growth. Contention is not the driver (write volume is tiny).
PARTITION_COUNT = 128

# Tenant-scoped data tables partitioned by HASH(tenant_id). The service tables
# (queue, active_queue_sessions) and the tenants registry are NOT partitioned.
PARTITIONED_TABLES = frozenset(
    {
        "workspaces",
        "peers",
        "sessions",
        "messages",
        "message_embeddings",
        "collections",
        "documents",
        "session_peers",
        "webhook_endpoints",
    }
)


def _qualified(table: Table) -> str:
    return f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'


def _create_hash_partitions(table: Table) -> None:
    """Create the N HASH partitions for a partitioned parent table."""
    schema_prefix = f'"{table.schema}".' if table.schema else ""
    for remainder in range(PARTITION_COUNT):
        op.execute(
            f'CREATE TABLE {schema_prefix}"{table.name}_p{remainder:03d}"'
            + f" PARTITION OF {_qualified(table)}"
            + f" FOR VALUES WITH (MODULUS {PARTITION_COUNT}, REMAINDER {remainder})"
        )


def upgrade() -> None:
    metadata = Base.metadata
    # Create tables in FK-dependency order (tenants first). A partitioned
    # parent's HASH partitions are created immediately after the parent.
    for table in metadata.sorted_tables:
        op.execute(CreateTable(table))
        if table.name in PARTITIONED_TABLES:
            _create_hash_partitions(table)
    # Create indexes once every partition exists, so each partitioned index
    # cascades onto all partitions.
    for table in metadata.sorted_tables:
        for index in table.indexes:
            op.execute(CreateIndex(index))


def downgrade() -> None:
    metadata = Base.metadata
    # Reverse dependency order; CASCADE drops each partitioned table's partitions
    # (and any dependent FKs) along with the parent.
    for table in reversed(metadata.sorted_tables):
        op.execute(f"DROP TABLE IF EXISTS {_qualified(table)} CASCADE")
