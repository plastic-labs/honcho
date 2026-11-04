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

revision: str = "a1e7ce5396d5"
down_revision: str | None = "b7d2e4f19c03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

MESSAGE_TEXT_MATCHES = "message_text_matches"


def message_text_matches_ddl(schema: str) -> str:
    return f"""
CREATE OR REPLACE FUNCTION "{schema}".{MESSAGE_TEXT_MATCHES}(
    p_tenant text, p_workspace text, p_query text, p_pattern text, p_use_fts boolean
) RETURNS SETOF bigint
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    match text := 'm.content ILIKE $2 ESCAPE ' || quote_literal('\\');
BEGIN
    IF p_tenant IS NULL OR p_tenant IS DISTINCT FROM current_setting('app.tenant', true)
    THEN
        RETURN;
    END IF;
    IF p_use_fts THEN
        match := 'to_tsvector(''english'', m.content)'
            || ' @@ plainto_tsquery(''english'', $3) OR ' || match;
    END IF;
    RETURN QUERY EXECUTE 'SELECT m.id FROM "{schema}".messages AS m'
        || ' WHERE m.tenant_id = ' || quote_literal(p_tenant)
        || ' AND m.workspace_name = $1 AND (' || match || ')'
        USING p_workspace, p_pattern, p_query;
END
$$
"""  # nosec B608


def upgrade() -> None:
    op.execute(message_text_matches_ddl(schema))


def downgrade() -> None:
    op.execute(
        f'DROP FUNCTION IF EXISTS "{schema}".{MESSAGE_TEXT_MATCHES}'
        + "(text, text, text, text, boolean)"
    )
