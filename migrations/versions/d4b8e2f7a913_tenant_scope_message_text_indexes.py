"""lead the message text-search indexes with tenant_id and workspace_name

messages is HASH(tenant_id)-partitioned, so one partition holds a slice of
every tenant. Message search (src/utils/search.py::message_text_condition, and
the message_text_matches function it calls under RLS) prunes to the tenant's
partition and then matches content with full-text search OR'd with
``content ILIKE '%q%'``. The two GIN indexes that served those matches,
ix_messages_content_gin and ix_messages_content_trgm, had only the text column,
so each bitmap they returned held the matches of every tenant in the partition
and the tenant and workspace predicates filtered rows afterwards. A search's
cost grew with the messages of every tenant sharing the partition, not with
the searching tenant's own.

This revision replaces them with ix_messages_tenant_content_fts
``(tenant_id, workspace_name, to_tsvector('english', content))`` and
ix_messages_tenant_content_trgm ``(tenant_id, workspace_name, content
gin_trgm_ops)``. btree_gin lets a GIN index carry those scalar columns; with
them the bitmap is ANDed down to the workspace before the heap is touched. A
GIN index serves any subset of its columns, so the single-tenant path (no
tenant predicate) uses them too, and the old indexes are dropped.

btree_gin and pg_trgm are created in migrations/env.py. Where the migration
role can't create btree_gin, nothing changes and the old indexes stay. Where
pg_trgm is missing, only the full-text index is replaced.

The builds hold no long write lock, on plain or partitioned tables (see
migrations.utils.create_index_online); each partition's index is built
CONCURRENTLY and attached. On a large database each build takes minutes per
partition, and the index is about 1.5x the size of messages for the trigram
one. To build them ahead of a release, run the same steps by hand under the
same names; this revision then finds them done and only drops the old indexes.

Revision ID: d4b8e2f7a913
Revises: a1e7ce5396d5
Create Date: 2026-10-30

"""

import logging
from collections.abc import Sequence

from alembic import op
from sqlalchemy import text

from migrations.utils import create_index_online, drop_index_online

# revision identifiers, used by Alembic.
revision: str = "d4b8e2f7a913"
down_revision: str | None = "a1e7ce5396d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "messages"

# (new name, new columns, old name, old columns, extension beyond btree_gin)
INDEXES: list[tuple[str, str, str, str, str | None]] = [
    (
        "ix_messages_tenant_content_fts",
        "tenant_id, workspace_name, to_tsvector('english', content)",
        "ix_messages_content_gin",
        "to_tsvector('english', content)",
        None,
    ),
    (
        "ix_messages_tenant_content_trgm",
        "tenant_id, workspace_name, content gin_trgm_ops",
        "ix_messages_content_trgm",
        "content gin_trgm_ops",
        "pg_trgm",
    ),
]

logger = logging.getLogger("alembic")


def _extension_installed(extension: str) -> bool:
    return bool(
        op.get_bind().scalar(
            text("SELECT 1 FROM pg_extension WHERE extname = :name"),
            {"name": extension},
        )
    )


def _skip(index: str, extension: str) -> None:
    logger.warning(
        "%s is not installed; skipping %s. Message search keeps its current"
        + " index. To add it later, install %s as a privileged role and build"
        + " the index with the steps in migrations.utils.create_index_online",
        extension,
        index,
        extension,
    )


def upgrade() -> None:
    if not _extension_installed("btree_gin"):
        for new, _, _, _, _ in INDEXES:
            _skip(new, "btree_gin")
        return
    for new, new_columns, old, _, extension in INDEXES:
        if extension and not _extension_installed(extension):
            _skip(new, extension)
            continue
        create_index_online(new, TABLE, new_columns, using="gin")
        drop_index_online(old, TABLE)


def downgrade() -> None:
    for new, _, old, old_columns, extension in reversed(INDEXES):
        if not extension or _extension_installed(extension):
            create_index_online(old, TABLE, old_columns, using="gin")
        drop_index_online(new, TABLE)
