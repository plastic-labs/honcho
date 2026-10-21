"""index the reads that grow with one tenant's data

Three API reads had no index for their filter and order. Each one read all of a
workspace, a collection or a session to return a page:

* conclusions/list (crud.get_documents_with_filters): a workspace's live
  documents, newest first, and their count for the page total. The page walked
  the partition-wide created_at index past other tenants' rows; the count read
  every document of the workspace from the heap.
* peer context and representation (crud.query_documents_most_derived,
  RepresentationManager._query_documents_most_derived): a collection's live
  documents by times_derived DESC, created_at DESC, id, sorted in full for a LIMIT.
* session context (crud.get_messages_id_range with a token limit) and the
  deriver's batch window: a session's messages by id with a running sum of
  token_count. INCLUDE (token_count) makes that sum index-only.

All three lead with tenant_id, like the indexes of e5fe7f8bcf62. Under RLS
the planner uses tenant_id = current_setting('app.tenant') as the first index
condition. Single-tenant queries have no tenant_id condition, so there these
indexes give no order until those queries bind tenant_id too.

The builds hold no long write lock, on plain or partitioned tables (see
migrations.utils.create_index_online). On a large database each one takes
minutes per partition. To build them ahead of a release, run the same steps by
hand; this revision then finds them done.

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
        "token_count",
        None,
    ),
]


def upgrade() -> None:
    for name, table, columns, include, where in INDEXES:
        create_index_online(name, table, columns, include=include, where=where)


def downgrade() -> None:
    for name, table, *_ in reversed(INDEXES):
        drop_index_online(name, table)
