"""index tenant-scoped reads and retire the indexes they replace

This revision adds the indexes below. Then it drops the indexes that they make
unnecessary. Each read that a new index serves also has an explicit tenant_id
predicate (models.scoped_tenant_id). Thus the index gives the read its order
with MULTI_TENANT on (RLS) and off.

Reconciler pending claim:

* ix_documents_pending_claim and ix_message_embeddings_pending_claim hold only
  the rows with sync_state = 'pending', in last_sync_at NULLS FIRST order.
* The documents claim orders by last_sync_at NULLS FIRST with a LIMIT. The full
  (sync_state, last_sync_at) indexes are ASC NULLS LAST, so the plan sorts every
  pending row and the LIMIT cannot stop early. With the partial index, the plan
  is a Merge Append that stops at the LIMIT. On a hash-partitioned replica with
  1M documents: 822 -> 88 buffers with 2,000 pending rows, and 3,117 -> 105
  buffers with 100,000 pending rows.
* The message_embeddings claim groups by message_id, so an index cannot stop
  its LIMIT early. Its partial index serves the other pending reads of that
  table: has_pending_work, the pending-count gauge and embed_now.
* Every sync_state predicate in src/ is sync_state = 'pending'. Thus this
  revision drops the two full (sync_state, last_sync_at) indexes. A full index
  gets an entry when a row is inserted and another one when the row is marked
  synced. A partial index gets only the first entry.

Lists in created_at order:

* ix_sessions_tenant_workspace_created and ix_peers_tenant_workspace_created
  serve the session, peer and scope lists of one workspace. They replace
  ix_*_tenant_workspace (the same first columns) and ix_*_created_at.
* ix_documents_tenant_collection_created adds created_at to
  ix_documents_tenant_collection, for the newest documents of one collection.
* This revision drops ix_documents_created_at. ix_documents_workspace_recent
  (revision 647b773f05ac) and the collection index serve the reads that used it.
* This revision drops ix_message_embeddings_created_at. No query orders or
  filters message_embeddings by created_at.

Other reads:

* ix_documents_tenant_session serves the reads of the documents of one session:
  session delete and scope backfill.
* ix_documents_legacy_sources_tenant replaces
  ix_documents_legacy_sources_pending with (tenant_id, id), the backfill's
  order. The predicate does not change.
* ix_queue_tenant_workspace replaces ix_queue_tenant_id. ix_queue_workspace_name
  stays for MULTI_TENANT off, where the queue reads have no tenant_id predicate.
* This revision drops ix_messages_session_lookup. ix_messages_session_tokens
  (revision 647b773f05ac) serves its reads.
* ix_documents_deleted_at becomes partial (WHERE deleted_at IS NOT NULL).
  Revision 119a52b73c60 made it partial, but a database built from the models
  has a full index. This revision builds the partial index under a temporary
  name. Then it drops the full index and renames the partial index.

The revision makes each new index before it drops the index that the new one
replaces. The builds do not hold a long write lock, on plain or partitioned
tables (see migrations.utils.create_index_online). To build the indexes before
a release, do the same steps manually. This revision then finds them done.

Revision ID: a3f9c2e81d47
Revises: 647b773f05ac
Create Date: 2026-10-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    create_index_online,
    drop_index_online,
    get_schema,
    index_valid,
)

# revision identifiers, used by Alembic.
revision: str = "a3f9c2e81d47"
down_revision: str | None = "647b773f05ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_SOURCES = (
    "source_ids IS NOT NULL OR internal_metadata ?| ARRAY['source_ids', 'premise_ids']"
)

# (name, table, columns, include, where): the same as in src/models.py.
NEW: list[tuple[str, str, str, str | None, str | None]] = [
    (
        "ix_documents_pending_claim",
        "documents",
        "last_sync_at NULLS FIRST",
        None,
        "sync_state = 'pending'",
    ),
    (
        "ix_message_embeddings_pending_claim",
        "message_embeddings",
        "last_sync_at NULLS FIRST",
        None,
        "sync_state = 'pending'",
    ),
    (
        "ix_sessions_tenant_workspace_created",
        "sessions",
        "tenant_id, workspace_name, created_at, id",
        None,
        None,
    ),
    (
        "ix_peers_tenant_workspace_created",
        "peers",
        "tenant_id, workspace_name, created_at, id",
        None,
        None,
    ),
    (
        "ix_documents_tenant_collection_created",
        "documents",
        "tenant_id, observer, observed, workspace_name, created_at",
        None,
        None,
    ),
    (
        "ix_documents_tenant_session",
        "documents",
        "tenant_id, workspace_name, session_name",
        None,
        None,
    ),
    (
        "ix_documents_legacy_sources_tenant",
        "documents",
        "tenant_id, id",
        None,
        LEGACY_SOURCES,
    ),
    (
        "ix_queue_tenant_workspace",
        "queue",
        "tenant_id, workspace_name",
        None,
        None,
    ),
]

# The indexes that this revision drops, as the earlier revisions made them.
# (name, table, columns, include, where)
RETIRED: list[tuple[str, str, str, str | None, str | None]] = [
    (
        "ix_documents_sync_state_last_sync_at",
        "documents",
        "sync_state, last_sync_at",
        None,
        None,
    ),
    (
        "ix_message_embeddings_sync_state_last_sync_at",
        "message_embeddings",
        "sync_state, last_sync_at",
        None,
        None,
    ),
    (
        "ix_sessions_tenant_workspace",
        "sessions",
        "tenant_id, workspace_name",
        None,
        None,
    ),
    ("ix_sessions_created_at", "sessions", "created_at", None, None),
    ("ix_peers_tenant_workspace", "peers", "tenant_id, workspace_name", None, None),
    ("ix_peers_created_at", "peers", "created_at", None, None),
    (
        "ix_documents_tenant_collection",
        "documents",
        "tenant_id, observer, observed, workspace_name",
        None,
        None,
    ),
    ("ix_documents_created_at", "documents", "created_at", None, None),
    (
        "ix_message_embeddings_created_at",
        "message_embeddings",
        "created_at",
        None,
        None,
    ),
    (
        "ix_documents_legacy_sources_pending",
        "documents",
        "id",
        None,
        LEGACY_SOURCES,
    ),
    ("ix_queue_tenant_id", "queue", "tenant_id", None, None),
    (
        "ix_messages_session_lookup",
        "messages",
        "tenant_id, session_name, id",
        "created_at",
        None,
    ),
]

DELETED_AT = "ix_documents_deleted_at"
# The partial index has this name until this revision drops the full index.
DELETED_AT_TEMP = "ix_documents_deleted_at_partial"
DELETED_AT_WHERE = "deleted_at IS NOT NULL"


def _is_partial(name: str) -> bool:
    """True if the index has a WHERE clause."""
    return bool(
        op.get_bind().scalar(
            sa.text(
                "SELECT i.indpred IS NOT NULL FROM pg_index i"
                + " WHERE i.indexrelid = to_regclass(:name)"
            ),
            {"name": f'"{get_schema()}".{name}'},
        )
    )


def _make_deleted_at_partial() -> None:
    """Replace a full ix_documents_deleted_at with the partial index.

    The cleanup reads have an index at each step: the full index stays until
    the partial index is valid. Each step is safe to run again.
    """
    # Done. Do not clean up DELETED_AT_TEMP here: after the rename, the partition
    # indexes keep the temporary name and belong to DELETED_AT.
    if index_valid(DELETED_AT) and _is_partial(DELETED_AT):
        return
    create_index_online(
        DELETED_AT_TEMP, "documents", "deleted_at", where=DELETED_AT_WHERE
    )
    drop_index_online(DELETED_AT, "documents")
    op.execute(
        f'ALTER INDEX "{get_schema()}"."{DELETED_AT_TEMP}" RENAME TO "{DELETED_AT}"'
    )


def upgrade() -> None:
    for name, table, columns, include, where in NEW:
        create_index_online(name, table, columns, include=include, where=where)
    _make_deleted_at_partial()
    for name, table, *_ in RETIRED:
        drop_index_online(name, table)


def downgrade() -> None:
    for name, table, columns, include, where in RETIRED:
        create_index_online(name, table, columns, include=include, where=where)
    # ix_documents_deleted_at stays partial: revision 119a52b73c60 made it so.
    for name, table, *_ in reversed(NEW):
        drop_index_online(name, table)
