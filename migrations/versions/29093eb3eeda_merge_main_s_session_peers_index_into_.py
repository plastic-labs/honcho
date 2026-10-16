"""merge main's session_peers index into the shared-tenants chain

Revision ID: 29093eb3eeda
Revises: 43d77d0c5846, b8d2f4a6c9e1
Create Date: 2026-10-01

A merge revision with no schema change. main added b8d2f4a6c9e1 (index
session_peers by peer) off a7c3e9f1b2d4 while the shared-tenants chain grew
past that revision, so merging main into feat/shared-tenants left two heads.

The two branches commute. b8d2f4a6c9e1 creates a plain (non-CONCURRENTLY)
index guarded by index_exists. The tenant-id primitive (e5fe7f8bcf62) alters
session_peers in place (it adds a column and replaces the primary key and
foreign keys) and never recreates the table. So the index survives in either
order: a database that ran b8d2f4a6c9e1 first (one already on main) and a
fresh one that runs the shared-tenants chain first both reach this head with
the index. The upgrade path is unchanged for databases on either branch.
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "29093eb3eeda"
down_revision: str | Sequence[str] | None = ("43d77d0c5846", "b8d2f4a6c9e1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
