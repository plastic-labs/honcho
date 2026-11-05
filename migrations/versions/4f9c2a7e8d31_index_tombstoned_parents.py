"""Index tombstoned peers and sessions for tenant-scoped lifecycle reads.

Expand only: retain all existing indexes and leave deletion inactive. Online
builds support both plain deployed tables and hash-partitioned tenant tables,
including resuming interrupted builds. No data or rollback floor changes.

Revision ID: 4f9c2a7e8d31
Revises: a7e3c1d9f2b4
Create Date: 2026-11-05
"""

from collections.abc import Sequence

from migrations.utils import create_index_online, drop_index_online

revision: str = "4f9c2a7e8d31"
down_revision: str | None = "a7e3c1d9f2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEXES = (
    ("ix_peers_tenant_workspace_deleted", "peers"),
    ("ix_sessions_tenant_workspace_deleted", "sessions"),
)


def upgrade() -> None:
    """Build only the partial parent-tombstone indexes."""
    for name, table in INDEXES:
        create_index_online(
            name,
            table,
            "tenant_id, workspace_name, name",
            where="deleted_at IS NOT NULL",
        )


def downgrade() -> None:
    """Remove only these indexes for development reversal."""
    for name, table in reversed(INDEXES):
        drop_index_online(name, table)
