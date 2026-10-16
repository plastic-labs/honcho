"""merge main's messages trigram index into the shared-tenants chain

Revision ID: c075274557cd
Revises: 29093eb3eeda, 3e7a1c9d5b20
Create Date: 2026-10-15

A merge revision with no schema change. main added 3e7a1c9d5b20 (a pg_trgm GIN
index on messages.content) off b8d2f4a6c9e1 while the shared-tenants chain grew
past that revision, so merging main into feat/shared-tenants left two heads.

The two branches commute. 3e7a1c9d5b20 builds the index guarded by
if_not_exists and touches nothing else. The tenant-id primitive (e5fe7f8bcf62)
alters messages in place (it adds a column and replaces the primary key) and
never recreates the table or rewrites content. So a database that ran
3e7a1c9d5b20 first (one already on main) and a fresh one that runs the
shared-tenants chain first both reach this head with the index. The upgrade
path is unchanged for databases on either branch.
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "c075274557cd"
down_revision: str | Sequence[str] | None = ("29093eb3eeda", "3e7a1c9d5b20")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
