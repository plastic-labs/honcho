"""add the SECURITY DEFINER function behind message text search under RLS

Revision ID: a1e7ce5396d5
Revises: b7d2e4f19c03
Create Date: 2026-10-27

Message search filters messages.content with ``@@`` (full-text) and ``ILIKE``.
Neither operator is leakproof, so under row-level security Postgres evaluates
them only after the policy and never as an index condition: for the app role
every search evaluated to_tsvector over the whole workspace, and grep scanned
the partition. The function this revision creates
runs as its owner, which bypasses RLS, so the GIN indexes on messages.content
serve it. It pins the tenant it reads to the bound app.tenant and returns only
message ids; the caller's RLS-scoped statement reads those rows by key.

The role that runs this revision owns the function, so it must have BYPASSRLS.
validate_tenant_isolation refuses to boot
under MULTI_TENANT when the owner cannot bypass RLS; ``ALTER FUNCTION ...
OWNER TO <a BYPASSRLS role>`` repairs it. With MULTI_TENANT off the function
is never called.
"""

from collections.abc import Sequence

from alembic import op

from migrations.utils import get_schema
from src.models import MESSAGE_TEXT_MATCHES, message_text_matches_ddl

revision: str = "a1e7ce5396d5"
down_revision: str | None = "b7d2e4f19c03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    op.execute(message_text_matches_ddl(schema))


def downgrade() -> None:
    op.execute(
        f'DROP FUNCTION IF EXISTS "{schema}".{MESSAGE_TEXT_MATCHES}'
        + "(text, text, text, text, boolean)"
    )
