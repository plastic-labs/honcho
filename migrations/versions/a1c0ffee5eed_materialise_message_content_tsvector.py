"""materialise message content tsvector for ts_rank

A GIN index accelerates full-text *matching*, but `ts_rank` needs the tsvector
VALUE, which a GIN index cannot return. The previous query form therefore
re-derived `to_tsvector('english', content)` for every candidate row on every
search, purely to order the results.

Measured on a 230 MB / 33,098-message workspace (PostgreSQL 15 + pgvector):

    COUNT(*)                                     6 ms
    full scan reading all content            1,239 ms
    to_tsvector(content) @@ tsquery  (GIN)      25 ms
    ts_rank(to_tsvector(content), ...)      48,813 ms   <-- entire cost

Against a stored column the same workload runs in 1,593 ms, and the real
query shape (WHERE + ORDER BY ts_rank + LIMIT) drops from 48,813 ms to 872 ms
- roughly 56x. Storage cost is ~127 MB on that workspace (230 MB -> 357 MB
including the index).

The backfill is a one-time table rewrite (~73 s for 33k rows on a 2013 i5) and
holds an ACCESS EXCLUSIVE lock for its duration. Schedule accordingly on large
deployments.

Revision ID: a1c0ffee5eed
Revises: e4eba9cfaa6f
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1c0ffee5eed"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # STORED generated column: computed once on write, read directly by ts_rank.
    op.execute(
        """
        ALTER TABLE messages
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_messages_content_tsv_gin
        ON messages USING gin (content_tsv)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_messages_content_tsv_gin")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS content_tsv")
