"""index session_peers by tenant

ix_session_peers_tenant_workspace_peer replaces ix_session_peers_workspace_peer.
The new index has tenant_id first. Thus a read of the sessions of one peer stays
in one tenant. The old index started with workspace_name, so its range also held
the rows of other tenants that have a workspace and a peer with the same names.

Each session_peers read that filters by workspace now has an explicit tenant_id
predicate (models.scoped_tenant_id). Thus the index gives the read its order with
MULTI_TENANT on (RLS) and off.

The revision makes the new index before it drops the old one. The builds do not
hold a long write lock (see migrations.utils.create_index_online).

Revision ID: b3c890d417d1
Revises: 0c67be41049b
Create Date: 2026-10-22

"""

from collections.abc import Sequence

from migrations.utils import create_index_online, drop_index_online

# revision identifiers, used by Alembic.
revision: str = "b3c890d417d1"
down_revision: str | None = "0c67be41049b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW = (
    "ix_session_peers_tenant_workspace_peer",
    "tenant_id, workspace_name, peer_name, session_name",
)
OLD = ("ix_session_peers_workspace_peer", "workspace_name, peer_name, session_name")


def upgrade() -> None:
    create_index_online(NEW[0], "session_peers", NEW[1])
    drop_index_online(OLD[0], "session_peers")


def downgrade() -> None:
    create_index_online(OLD[0], "session_peers", OLD[1])
    drop_index_online(NEW[0], "session_peers")
