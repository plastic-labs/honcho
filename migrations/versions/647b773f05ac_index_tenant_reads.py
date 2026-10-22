"""index the reads that grow with one tenant's data

Three reads that grow with a workspace, a collection or a session had no index
that serves them:

* conclusions/list (crud.get_documents_with_filters): a workspace's live
  documents, newest first, and their count for the page total. The page walked
  the partition-wide created_at index past other tenants' rows; the count read
  every document of the workspace from the heap.
* peer context and representation (RepresentationManager._query_documents_most_derived):
  a collection's live documents by times_derived DESC, created_at DESC, id,
  sorted in full for a LIMIT.
* session context (crud.get_messages_id_range with a token limit) and the
  deriver's batch window: a session's messages by id with a running sum of
  token_count. ix_messages_session_lookup serves the order but not the sum, so
  each message's token_count came from the heap. The new index keys on
  workspace_name too and carries (created_at, token_count), so it covers every
  use of ix_messages_session_lookup; that index can be dropped once nothing
  reads it. The session-context sum is index-only; the deriver also reads
  peer_name from the heap.

All three lead with tenant_id, like the other tenant-scoped lookup indexes in
src/models.py, and the reads they serve filter on tenant_id explicitly
(models.scoped_tenant_id: the bound tenant, or 'default' when MULTI_TENANT is
off). With tenant_id as the leading equality the rest of the key reads in
order. A query with no tenant_id condition gets no order from them.

The builds hold no long write lock, on plain or partitioned tables (see
migrations.utils.create_index_online). On a large database each one takes
minutes per partition. To build them ahead of a release, run the same steps by
hand under the same names; this revision then finds them done.

Revision ID: 647b773f05ac
Revises: c075274557cd
Create Date: 2026-10-21

"""

from collections.abc import Sequence

from migrations.utils import create_index_online, drop_index_online

# revision identifiers, used by Alembic.
revision: str = "647b773f05ac"
down_revision: str | None = "c075274557cd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (name, table, columns, include, where): the same as in src/models.py.
INDEXES: list[tuple[str, str, str, str | None, str | None]] = [
    (
        "ix_documents_workspace_recent",
        "documents",
        "tenant_id, workspace_name, created_at, id",
        None,
        "deleted_at IS NULL",
    ),
    (
        "ix_documents_collection_derived",
        "documents",
        "tenant_id, workspace_name, observer, observed, times_derived DESC, created_at DESC, id",
        None,
        "deleted_at IS NULL",
    ),
    (
        "ix_messages_session_tokens",
        "messages",
        "tenant_id, workspace_name, session_name, id",
        "created_at, token_count",
        None,
    ),
]


def upgrade() -> None:
    for name, table, columns, include, where in INDEXES:
        create_index_online(name, table, columns, include=include, where=where)


def downgrade() -> None:
    for name, table, *_ in reversed(INDEXES):
        drop_index_online(name, table)
