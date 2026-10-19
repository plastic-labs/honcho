"""queue pending-work indexes and deletion generations

Adds partial expression indexes over the queue payload so unprocessed work can
be found by the peer, session or message it touches, and a deletion_generation
counter (INTEGER NOT NULL DEFAULT 0) on workspaces, sessions and collections.

Every index is partial on ``processed = false`` and on its expression being
non-null, so it holds only pending rows of the task types that carry the key:

- ix_queue_pending_observers: GIN on ``payload -> 'observers'``
  (representation fan-out, queried with ``?``)
- ix_queue_pending_observed: ``(workspace_name, payload ->> 'observed')``
  (representation, dream)
- ix_queue_pending_observer: ``(workspace_name, payload ->> 'observer')``
  (dream)
- ix_queue_pending_scope_peer: ``(workspace_name, payload ->> 'scope_peer')``
  (scope_backfill, scope_removal)
- ix_queue_pending_session_name: ``(workspace_name, payload ->> 'session_name')``
  (every session-scoped task; dream and scope tasks leave session_id NULL)

The columns are added first, in the migration transaction: a constant default
is a catalog-only change. The indexes are then built CONCURRENTLY in an
autocommit block with no lock_timeout and a BUILD_TIMEOUT statement_timeout,
and an INVALID leftover from a failed build is dropped and rebuilt. On a large
queue, prebuild them ahead of the release with the same statements; if_not_exists
then makes the build a no-op. The queue is not
partitioned in any lineage, so the concurrent path always applies. When env.py
runs on a caller-supplied connection (the alembic test pipeline), the indexes
are built with a plain, transactional CREATE INDEX.

Revision ID: a5f8da8f6d23
Revises: 3e7a1c9d5b20
Create Date: 2026-10-19

"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "a5f8da8f6d23"
down_revision: str | None = "3e7a1c9d5b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

BUILD_TIMEOUT = "60min"
GENERATION_COLUMN = "deletion_generation"
GENERATION_TABLES = ("workspaces", "sessions", "collections")

logger = logging.getLogger("alembic")


@dataclass(frozen=True, slots=True)
class _QueueIndex:
    name: str
    columns: tuple[str, ...]
    predicate: str
    using: str = "btree"


QUEUE_INDEXES = (
    _QueueIndex(
        name="ix_queue_pending_observers",
        columns=("(payload -> 'observers')",),
        predicate="processed = false AND (payload -> 'observers') IS NOT NULL",
        using="gin",
    ),
    _QueueIndex(
        name="ix_queue_pending_observed",
        columns=("workspace_name", "(payload ->> 'observed')"),
        predicate="processed = false AND (payload ->> 'observed') IS NOT NULL",
    ),
    _QueueIndex(
        name="ix_queue_pending_observer",
        columns=("workspace_name", "(payload ->> 'observer')"),
        predicate="processed = false AND (payload ->> 'observer') IS NOT NULL",
    ),
    _QueueIndex(
        name="ix_queue_pending_scope_peer",
        columns=("workspace_name", "(payload ->> 'scope_peer')"),
        predicate="processed = false AND (payload ->> 'scope_peer') IS NOT NULL",
    ),
    _QueueIndex(
        name="ix_queue_pending_session_name",
        columns=("workspace_name", "(payload ->> 'session_name')"),
        predicate="processed = false AND (payload ->> 'session_name') IS NOT NULL",
    ),
)


def _index_valid(connection: Connection, name: str) -> bool | None:
    """Return the index's validity, or None if it doesn't exist."""
    return connection.scalar(
        text(
            "SELECT i.indisvalid FROM pg_index i"
            + " JOIN pg_class c ON c.oid = i.indexrelid"
            + " JOIN pg_namespace n ON n.oid = c.relnamespace"
            + " WHERE c.relname = :name AND n.nspname = :schema"
        ),
        {"name": name, "schema": schema},
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


def _create_index(index: _QueueIndex, *, concurrently: bool) -> None:
    op.create_index(
        index.name,
        "queue",
        [text(column) for column in index.columns],
        schema=schema,
        postgresql_using=index.using,
        postgresql_where=text(index.predicate),
        postgresql_concurrently=concurrently,
        if_not_exists=True,
    )


def _drop_index(index: _QueueIndex, *, concurrently: bool) -> None:
    op.drop_index(
        index.name,
        table_name="queue",
        schema=schema,
        postgresql_concurrently=concurrently,
        if_exists=True,
    )


def _build_index(
    connection: Connection, index: _QueueIndex, *, concurrently: bool
) -> None:
    """Create the index, first dropping an INVALID leftover of a failed build."""
    if _index_valid(connection, index.name) is False:
        _drop_index(index, concurrently=concurrently)
    _create_index(index, concurrently=concurrently)


def upgrade() -> None:
    for table in GENERATION_TABLES:
        op.add_column(
            table,
            sa.Column(
                GENERATION_COLUMN,
                sa.Integer(),
                nullable=False,
                server_default=text("0"),
            ),
            schema=schema,
            if_not_exists=True,
        )

    connection = op.get_bind()
    if _caller_owns_transaction():
        for index in QUEUE_INDEXES:
            _build_index(connection, index, concurrently=False)
        return

    with op.get_context().autocommit_block(), _build_timeouts(connection):
        for index in QUEUE_INDEXES:
            logger.info(
                "Building %s concurrently; waits for open transactions on queue"
                + " and older snapshots anywhere in the database, up to %s",
                index.name,
                BUILD_TIMEOUT,
            )
            _build_index(connection, index, concurrently=True)


def downgrade() -> None:
    if _caller_owns_transaction():
        for index in QUEUE_INDEXES:
            _drop_index(index, concurrently=False)
    else:
        connection = op.get_bind()
        with op.get_context().autocommit_block(), _build_timeouts(connection):
            for index in QUEUE_INDEXES:
                _drop_index(index, concurrently=True)

    for table in GENERATION_TABLES:
        op.drop_column(table, GENERATION_COLUMN, schema=schema, if_exists=True)
