"""index messages.content with pg_trgm

Workspace and session search match messages with full-text search OR'd with
``content ILIKE '%q%'`` (src/utils/search.py::_fulltext_search). The FTS side
is covered by ix_messages_content_gin, but the leading-wildcard ILIKE has no
usable index, so the OR forces a sequential scan of the workspace's messages
on every search. Queries with punctuation take the ILIKE-only branch and scan
the same way. A trigram GIN index makes the ILIKE side indexable, so the
planner can BitmapOr the two indexes and visit only matching rows.

The index is built CONCURRENTLY so message writes are not blocked during the
build. That cannot run inside a transaction, so it runs in an autocommit
block, with lock_timeout and statement_timeout lifted for the build: the
migration connection carries a 1s lock_timeout and a 5 minute
statement_timeout, and a concurrent build waits for every open transaction on
the table before it starts. A build that fails part-way leaves an INVALID
index behind, which is dropped and rebuilt on the next run.

A caller that supplies its own connection (config.attributes["connection"],
as the alembic test pipeline does) owns the surrounding transaction, so an
autocommit block can't run there; that path builds the index with a plain,
transactional CREATE INDEX instead.

pg_trgm is created in migrations/env.py. Where the migration role cannot
create extensions (managed Postgres without the privilege), the index is
skipped with a warning and search keeps working without it.

Revision ID: 3e7a1c9d5b20
Revises: b8d2f4a6c9e1
Create Date: 2026-10-08

"""

import logging
from collections.abc import Sequence

from alembic import op
from sqlalchemy import Connection, text

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "3e7a1c9d5b20"
down_revision: str | None = "b8d2f4a6c9e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

INDEX_NAME = "ix_messages_content_trgm"

logger = logging.getLogger(__name__)


def _trgm_installed(connection: Connection) -> bool:
    return bool(
        connection.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))
    )


def _index_valid(connection: Connection) -> bool | None:
    """Return the index's validity, or None if it doesn't exist."""
    return connection.scalar(
        text(
            "SELECT i.indisvalid FROM pg_index i"
            + " JOIN pg_class c ON c.oid = i.indexrelid"
            + " JOIN pg_namespace n ON n.oid = c.relnamespace"
            + " WHERE c.relname = :name AND n.nspname = :schema"
        ),
        {"name": INDEX_NAME, "schema": schema},
    )


def _caller_owns_transaction() -> bool:
    """Mirror migrations/env.py: a supplied connection means an external transaction."""
    config = op.get_context().config
    return config is not None and config.attributes.get("connection") is not None


def upgrade() -> None:
    connection = op.get_bind()

    if not _trgm_installed(connection):
        logger.warning(
            "pg_trgm is not installed; skipping %s. Message search still works"
            + " but scans the table. Install pg_trgm as a privileged role and"
            + " re-run this migration to add the index.",
            INDEX_NAME,
        )
        return

    if _caller_owns_transaction():
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME}"
            + f' ON "{schema}".messages USING gin (content gin_trgm_ops)'
        )
        return

    with op.get_context().autocommit_block():
        lock_timeout = connection.scalar(text("SHOW lock_timeout"))
        statement_timeout = connection.scalar(text("SHOW statement_timeout"))
        connection.execute(text("SET lock_timeout = 0"))
        connection.execute(text("SET statement_timeout = 0"))
        try:
            if _index_valid(connection) is False:
                op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{schema}".{INDEX_NAME}')
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME}"
                + f' ON "{schema}".messages USING gin (content gin_trgm_ops)'
            )
        finally:
            connection.execute(
                text("SELECT set_config('lock_timeout', :value, false)"),
                {"value": lock_timeout},
            )
            connection.execute(
                text("SELECT set_config('statement_timeout', :value, false)"),
                {"value": statement_timeout},
            )


def downgrade() -> None:
    if _caller_owns_transaction():
        op.execute(f'DROP INDEX IF EXISTS "{schema}".{INDEX_NAME}')
        return

    with op.get_context().autocommit_block():
        op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{schema}".{INDEX_NAME}')
