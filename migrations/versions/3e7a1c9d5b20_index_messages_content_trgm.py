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
block. The migration connection carries a 1s lock_timeout (from
src.migrate.run_with_lock_retry) and a 5 minute statement_timeout (from
env.py), and a concurrent build waits for every open transaction on the table
before it starts, so both are replaced for the build: no lock_timeout, and a
statement_timeout of BUILD_TIMEOUT so a stuck idle-in-transaction session
fails the migration instead of hanging it. A build that fails part-way leaves
an INVALID index behind, which is dropped and rebuilt on the next run.

When env.py runs on a caller-supplied connection (the alembic test pipeline),
the caller owns the surrounding transaction and an autocommit block can't
run, so that path builds the index with a plain, transactional CREATE INDEX.

pg_trgm is created in migrations/env.py. Where the migration role cannot
create extensions, the index is skipped with a warning and search keeps
working without it. The revision is still stamped, so the index then has to
be created by hand (the warning prints the statement); the model declares it
regardless.

Revision ID: 3e7a1c9d5b20
Revises: b8d2f4a6c9e1
Create Date: 2026-10-08

"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager

from alembic import op
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "3e7a1c9d5b20"
down_revision: str | None = "b8d2f4a6c9e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

INDEX_NAME = "ix_messages_content_trgm"
BUILD_TIMEOUT = "60min"

logger = logging.getLogger("alembic")


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
    """True when env.py is running inside a transaction the caller opened."""
    config = op.get_context().config
    return config is not None and not config.attributes.get("owns_transaction", True)


@contextmanager
def _build_timeouts(connection: Connection) -> Generator[None]:
    """Replace the session's lock/statement timeouts for a concurrent build."""
    lock_timeout = connection.scalar(text("SHOW lock_timeout"))
    statement_timeout = connection.scalar(text("SHOW statement_timeout"))
    connection.execute(text("SET lock_timeout = 0"))
    connection.execute(text(f"SET statement_timeout = '{BUILD_TIMEOUT}'"))
    try:
        yield
    finally:
        try:
            for name, value in (
                ("lock_timeout", lock_timeout),
                ("statement_timeout", statement_timeout),
            ):
                connection.execute(
                    text("SELECT set_config(:name, :value, false)"),
                    {"name": name, "value": value},
                )
        except DBAPIError:
            # A dropped connection can't be restored; let the build's own
            # error surface instead of this one.
            if not connection.invalidated:
                raise


def upgrade() -> None:
    connection = op.get_bind()

    if not _trgm_installed(connection):
        logger.warning(
            "pg_trgm is not installed; skipping %s. Message search still works"
            + " but scans the table. To add the index later, install pg_trgm as"
            + " a privileged role and run: CREATE INDEX CONCURRENTLY %s"
            + ' ON "%s".messages USING gin (content gin_trgm_ops)',
            INDEX_NAME,
            INDEX_NAME,
            schema,
        )
        return

    if _caller_owns_transaction():
        op.create_index(
            INDEX_NAME,
            "messages",
            ["content"],
            schema=schema,
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
            if_not_exists=True,
        )
        return

    with op.get_context().autocommit_block(), _build_timeouts(connection):
        if _index_valid(connection) is False:
            op.drop_index(
                INDEX_NAME,
                table_name="messages",
                schema=schema,
                postgresql_concurrently=True,
                if_exists=True,
            )
        logger.info(
            "Building %s concurrently; waits for open transactions on messages",
            INDEX_NAME,
        )
        op.create_index(
            INDEX_NAME,
            "messages",
            ["content"],
            schema=schema,
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
            postgresql_concurrently=True,
            if_not_exists=True,
        )


def downgrade() -> None:
    if _caller_owns_transaction():
        op.drop_index(INDEX_NAME, table_name="messages", schema=schema, if_exists=True)
        return

    connection = op.get_bind()
    with op.get_context().autocommit_block(), _build_timeouts(connection):
        op.drop_index(
            INDEX_NAME,
            table_name="messages",
            schema=schema,
            postgresql_concurrently=True,
            if_exists=True,
        )
