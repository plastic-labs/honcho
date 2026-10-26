"""keep explicit-document counters on collections for the dream-due scan

Revision ID: b7d2e4f19c03
Revises: b3c890d417d1
Create Date: 2026-10-23

The dream-due scan grouped every explicit document to find collections due a
dream. This revision adds explicit_count and newest_explicit_at to collections,
which the document writers keep current, backfills them once, and indexes the
due predicate (see src/models.py) so the scan reads only the due collections.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    build_timeouts,
    caller_owns_transaction,
    column_exists,
    create_index_online,
    drop_index_online,
    get_schema,
)

revision: str = "b7d2e4f19c03"
down_revision: str | None = "b3c890d417d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

DOCUMENTS_SINCE = (
    "explicit_count - COALESCE("
    + "((internal_metadata -> 'dream') ->> 'last_dream_document_count')::integer, 0)"
)
BACKLOG = (
    "COALESCE(((internal_metadata -> 'dream') ->> 'work_list_backlog')::integer, 0)"
)


def upgrade() -> None:
    if not column_exists("collections", "explicit_count"):
        op.add_column(
            "collections",
            sa.Column(
                "explicit_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            schema=schema,
        )
    if not column_exists("collections", "newest_explicit_at"):
        op.add_column(
            "collections",
            sa.Column("newest_explicit_at", sa.DateTime(timezone=True), nullable=True),
            schema=schema,
        )
    _backfill()
    create_index_online(
        "ix_collections_dream_due",
        "collections",
        f"({DOCUMENTS_SINCE}), newest_explicit_at",
    )
    create_index_online(
        "ix_collections_dream_backlog",
        "collections",
        "newest_explicit_at",
        where=f"{BACKLOG} > 0",
    )


def _backfill_sql(collections: str, documents: str) -> str:
    return f"""
        UPDATE {collections} c
        SET explicit_count = s.n, newest_explicit_at = s.newest
        FROM (
            SELECT tenant_id, workspace_name, observer, observed,
                   count(*) AS n, max(created_at) AS newest
            FROM {documents}
            WHERE level = 'explicit'
            GROUP BY 1, 2, 3, 4
        ) s
        WHERE c.tenant_id = s.tenant_id
          AND c.workspace_name = s.workspace_name
          AND c.observer = s.observer
          AND c.observed = s.observed
        """


def _partitions_by_bound(table: str) -> dict[str, str]:
    rows = op.get_bind().execute(
        sa.text(
            "SELECT pg_get_expr(c.relpartbound, c.oid), c.relname FROM pg_inherits h"
            + " JOIN pg_class c ON c.oid = h.inhrelid"
            + " WHERE h.inhparent = to_regclass(:table) AND NOT h.inhdetachpending"
            + " ORDER BY c.relname"
        ),
        {"table": f'"{schema}"."{table}"'},
    )
    return {bound: name for bound, name in rows}


def _backfill() -> None:
    collections = f'"{schema}"."collections"'
    documents = f'"{schema}"."documents"'
    if caller_owns_transaction():
        op.execute(_backfill_sql(collections, documents))
        return
    collection_partitions = _partitions_by_bound("collections")
    if not collection_partitions:
        with build_timeouts():
            op.execute(_backfill_sql(collections, documents))
        return
    document_partitions = _partitions_by_bound("documents")
    with op.get_context().autocommit_block(), build_timeouts():
        for bound, partition in collection_partitions.items():
            paired = document_partitions.get(bound)
            op.execute(
                _backfill_sql(
                    f'"{schema}"."{partition}"',
                    f'"{schema}"."{paired}"' if paired else documents,
                )
            )


def downgrade() -> None:
    drop_index_online("ix_collections_dream_backlog", "collections")
    drop_index_online("ix_collections_dream_due", "collections")
    op.drop_column("collections", "newest_explicit_at", schema=schema)
    op.drop_column("collections", "explicit_count", schema=schema)
