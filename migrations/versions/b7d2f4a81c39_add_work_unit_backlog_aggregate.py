"""add work_unit_backlog aggregate for the deriver claim

Revision ID: b7d2f4a81c39
Revises: e5fe7f8bcf62
Create Date: 2026-09-10

Adds ``work_unit_backlog`` — one row per pending work unit (tenant_id,
task_type, pending_count, total_tokens, oldest_created_at) — plus the
triggers on ``queue`` that maintain it, and backfills it from any rows
already pending so an instance upgrading mid-queue starts exact.

The claim path reads this table instead of re-aggregating the queue on
every poll. Rows exist only while a unit has unprocessed items; the
delete guard keys on row *existence* in the queue, never on the counter.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = "b7d2f4a81c39"
down_revision: str | None = "e5fe7f8bcf62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

# region ai
# Trigger design notes the SQL below encodes:
# - AFTER ROW triggers, so the recompute paths see the queue already reflecting
#   the change — the aggregate over remaining unprocessed rows is exact under
#   any deletion order (cascades delete messages and queue rows in either
#   order; a recompute never needs the deleted message row).
# - INSERT is the hot direction and takes the fast path: +1 / +tokens / LEAST,
#   with the token lookup a partition-pruned PK probe when the row carries a
#   tenant (flag-on) and a plain id probe otherwise (flag-off = unpartitioned
#   alembic schema, or the 1-partition OSS default).
# - Completion / delete / re-open recompute the whole row from the queue. A
#   recompute racing a concurrent enqueue (READ COMMITTED) can transiently
#   under-count tokens by the not-yet-visible row; the next completion
#   re-exactifies, and the DELETE guard re-checks queue membership in its own
#   later snapshot, so a live unit never loses its backlog row. pending_count
#   is bookkeeping, not a claim input.
# - The reconciler/dream dedup path inserts then rolls back on the partial
#   unique index; trigger effects roll back with the transaction, so the
#   dedup loser is never counted.
# endregion


def upgrade() -> None:
    if not table_exists("work_unit_backlog"):
        op.create_table(
            "work_unit_backlog",
            sa.Column("work_unit_key", sa.TEXT(), primary_key=True),
            sa.Column("tenant_id", sa.TEXT(), nullable=True),
            sa.Column("task_type", sa.TEXT(), nullable=False),
            sa.Column("pending_count", sa.Integer(), nullable=False),
            sa.Column(
                "total_tokens",
                sa.BigInteger(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("oldest_created_at", sa.DateTime(timezone=True), nullable=False),
            schema=schema,
        )
    if not index_exists("work_unit_backlog", "ix_work_unit_backlog_tenant_id"):
        op.create_index(
            "ix_work_unit_backlog_tenant_id",
            "work_unit_backlog",
            ["tenant_id"],
            schema=schema,
        )
    if not index_exists(
        "work_unit_backlog", "ix_work_unit_backlog_oldest_created_at_key"
    ):
        op.create_index(
            "ix_work_unit_backlog_oldest_created_at_key",
            "work_unit_backlog",
            ["oldest_created_at", "work_unit_key"],
            schema=schema,
        )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}._queue_item_token_count(
            tenant TEXT, msg_id BIGINT, item_task_type TEXT
        ) RETURNS BIGINT AS $$
        DECLARE
            t BIGINT;
        BEGIN
            IF item_task_type <> 'representation' OR msg_id IS NULL THEN
                RETURN 0;
            END IF;
            IF tenant IS NULL THEN
                SELECT token_count INTO t
                FROM {schema}.messages WHERE id = msg_id;
            ELSE
                SELECT token_count INTO t
                FROM {schema}.messages
                WHERE tenant_id = tenant AND id = msg_id;
            END IF;
            RETURN coalesce(t, 0);
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}._work_unit_backlog_recompute(
            key TEXT
        ) RETURNS void AS $$
        BEGIN
            INSERT INTO {schema}.work_unit_backlog AS wub
                (work_unit_key, tenant_id, task_type, pending_count,
                 total_tokens, oldest_created_at)
            SELECT q.work_unit_key,
                   max(q.tenant_id),
                   max(q.task_type),
                   count(*),
                   coalesce(sum(
                       {schema}._queue_item_token_count(
                           q.tenant_id, q.message_id, q.task_type
                       )
                   ), 0),
                   min(q.created_at)
            FROM {schema}.queue q
            WHERE q.work_unit_key = key AND NOT q.processed
            GROUP BY q.work_unit_key
            ON CONFLICT (work_unit_key) DO UPDATE SET
                pending_count = EXCLUDED.pending_count,
                total_tokens = EXCLUDED.total_tokens,
                oldest_created_at = EXCLUDED.oldest_created_at;
            -- Membership-based guard: only drop the row when the queue holds
            -- nothing unprocessed for this unit in THIS statement's snapshot.
            DELETE FROM {schema}.work_unit_backlog
            WHERE work_unit_key = key
              AND NOT EXISTS (
                  SELECT 1 FROM {schema}.queue q
                  WHERE q.work_unit_key = key AND NOT q.processed
              );
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}.work_unit_backlog_apply()
        RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.processed THEN
                    RETURN NULL;
                END IF;
                INSERT INTO {schema}.work_unit_backlog AS wub
                    (work_unit_key, tenant_id, task_type, pending_count,
                     total_tokens, oldest_created_at)
                VALUES (
                    NEW.work_unit_key,
                    NEW.tenant_id,
                    NEW.task_type,
                    1,
                    {schema}._queue_item_token_count(
                        NEW.tenant_id, NEW.message_id, NEW.task_type
                    ),
                    NEW.created_at
                )
                ON CONFLICT (work_unit_key) DO UPDATE SET
                    pending_count = wub.pending_count + 1,
                    total_tokens = wub.total_tokens + EXCLUDED.total_tokens,
                    oldest_created_at = LEAST(
                        wub.oldest_created_at, EXCLUDED.oldest_created_at
                    );
            ELSIF TG_OP = 'UPDATE' THEN
                IF OLD.processed IS DISTINCT FROM NEW.processed THEN
                    PERFORM {schema}._work_unit_backlog_recompute(
                        NEW.work_unit_key
                    );
                END IF;
            ELSIF TG_OP = 'DELETE' THEN
                IF NOT OLD.processed THEN
                    PERFORM {schema}._work_unit_backlog_recompute(
                        OLD.work_unit_key
                    );
                END IF;
            END IF;
            RETURN NULL;
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        DROP TRIGGER IF EXISTS trg_work_unit_backlog ON {schema}.queue;
        CREATE TRIGGER trg_work_unit_backlog
        AFTER INSERT OR UPDATE OF processed OR DELETE ON {schema}.queue
        FOR EACH ROW EXECUTE FUNCTION {schema}.work_unit_backlog_apply();
        """
    )

    # Backfill: an instance upgrading with pending work starts exact.
    op.execute(
        f"""
        INSERT INTO {schema}.work_unit_backlog
            (work_unit_key, tenant_id, task_type, pending_count,
             total_tokens, oldest_created_at)
        SELECT q.work_unit_key,
               max(q.tenant_id),
               max(q.task_type),
               count(*),
               coalesce(sum(
                   {schema}._queue_item_token_count(
                       q.tenant_id, q.message_id, q.task_type
                   )
               ), 0),
               min(q.created_at)
        FROM {schema}.queue q
        WHERE NOT q.processed
        GROUP BY q.work_unit_key
        ON CONFLICT (work_unit_key) DO NOTHING;
        """
    )


def downgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS trg_work_unit_backlog ON {schema}.queue")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}.work_unit_backlog_apply()")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}._work_unit_backlog_recompute(TEXT)")
    op.execute(
        f"DROP FUNCTION IF EXISTS {schema}._queue_item_token_count(TEXT, BIGINT, TEXT)"
    )
    if table_exists("work_unit_backlog"):
        op.drop_table("work_unit_backlog", schema=schema)
