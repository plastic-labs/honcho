"""index session_peers by peer

The peer-sessions list joins sessions to session_peers filtered by peer_name,
ordered by sessions.created_at with a small LIMIT. session_peers is only
indexed by its primary key (workspace_name, session_name, peer_name), so the
planner walks ix_sessions_created_at backward across every session in the
workspace and probes the PK per row until it finds enough matches. For a
peer with few or old sessions that is the whole table on every call.

This index lets the planner start from the peer's own rows and sort those.
session_name is included so the probe into sessions is index-only.

Revision ID: b8d2f4a6c9e1
Revises: a7c3e9f1b2d4
Create Date: 2026-09-22

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "b8d2f4a6c9e1"
down_revision: str | None = "a7c3e9f1b2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

INDEX_NAME = "ix_session_peers_workspace_peer"


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not index_exists("session_peers", INDEX_NAME, inspector):
        op.create_index(
            INDEX_NAME,
            "session_peers",
            ["workspace_name", "peer_name", "session_name"],
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if index_exists("session_peers", INDEX_NAME, inspector):
        op.drop_index(INDEX_NAME, table_name="session_peers", schema=schema)
