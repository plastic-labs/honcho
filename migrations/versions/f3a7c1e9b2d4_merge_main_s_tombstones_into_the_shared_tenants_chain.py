"""merge main's primitive tombstones into the shared-tenants chain

Revision ID: f3a7c1e9b2d4
Revises: a3f9c2e81d47, d8c4f6a2b901
Create Date: 2026-11-02

A merge revision with no schema change. main added a5f8da8f6d23 (queue
pending-work indexes plus a deletion_generation counter on workspaces,
sessions and collections) and d8c4f6a2b901 (a nullable deleted_at tombstone on
workspaces, peers, sessions and messages) off 3e7a1c9d5b20 while the
shared-tenants chain grew past that revision, so merging main into
feat/shared-tenants left two heads.

The two branches commute. Both main revisions add columns guarded by
if_not_exists and build indexes only on queue, which stays unpartitioned. The
tenant-id primitive (e5fe7f8bcf62) alters the primitive tables in place and
never recreates them, and the tenant index revisions (647b773f05ac,
a3f9c2e81d47) touch neither the new columns nor queue. So a database that ran
main's two revisions first and a fresh one that runs the shared-tenants chain
first both reach this head with the same columns and indexes.
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "f3a7c1e9b2d4"
down_revision: str | Sequence[str] | None = ("a3f9c2e81d47", "d8c4f6a2b901")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
