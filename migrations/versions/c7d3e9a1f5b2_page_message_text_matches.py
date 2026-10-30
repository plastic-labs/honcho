"""page message text matches and revoke PUBLIC execute on the search functions

Revision ID: c7d3e9a1f5b2
Revises: a1e7ce5396d5
Create Date: 2026-10-28

Two fixes to the SECURITY DEFINER function behind message search under RLS:

- ``message_text_matches`` returned every matching message id, unordered, so
  for a common word the caller pulled tens of thousands of ids and sorted them
  to keep ten. ``message_text_matches_page`` returns one ordered page (full-text
  rank, then newest first) after a keyset cursor, and the caller walks pages
  until its own filters have kept enough rows.
- Postgres grants EXECUTE on a new function to PUBLIC. A SECURITY DEFINER
  function reachable by every role lets a role with no SELECT on messages probe
  a tenant's content by binding that tenant itself. Both functions now revoke
  PUBLIC; grant EXECUTE to the app role explicitly (the startup validator checks
  it).

The old function stays so the previous release keeps working on this schema;
a later revision drops it.
"""

from collections.abc import Sequence

from alembic import op

from migrations.utils import get_schema
from src.models import (
    MESSAGE_TEXT_MATCHES,
    MESSAGE_TEXT_MATCHES_ARGS,
    message_text_matches_ddl,
)

revision: str = "c7d3e9a1f5b2"
down_revision: str | None = "a1e7ce5396d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

LEGACY = f'"{schema}".message_text_matches(text, text, text, text, boolean)'


def upgrade() -> None:
    for statement in message_text_matches_ddl(schema):
        op.execute(statement)
    op.execute(f"REVOKE ALL ON FUNCTION {LEGACY} FROM PUBLIC")


def downgrade() -> None:
    op.execute(
        f'DROP FUNCTION IF EXISTS "{schema}".{MESSAGE_TEXT_MATCHES}'
        + MESSAGE_TEXT_MATCHES_ARGS
    )
    op.execute(f"GRANT EXECUTE ON FUNCTION {LEGACY} TO PUBLIC")
