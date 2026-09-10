"""add work_unit_backlog aggregate for the deriver claim

Revision ID: b7d2f4a81c39
Revises: e5fe7f8bcf62
Create Date: 2026-09-10

Adds ``work_unit_backlog`` — one row per pending work unit (tenant_id,
task_type, pending_count, total_tokens, oldest_created_at) — plus the
triggers on ``queue`` that maintain it, and backfills it (deriving each
unit's tenant from its key prefix) so an instance upgrading mid-queue
starts exact.

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
# - INSERT is the hot direction and takes a row-level fast path: +1 / +tokens /
#   LEAST via upsert. The token lookup probes messages by (tenant_id, id) — the
#   partition-pruned composite PK — falling back to the single-tenant default
#   tenant when the queue row carries no tenant (an id-only probe cannot use
#   the composite PK as a search key and would scan).
# - Completion and deletion are statement-level triggers over transition
#   tables: one exact recompute per DISTINCT affected key, iterated in sorted
#   key order, instead of one per row (a batch completion previously ran the
#   same recompute N times; a bulk delete likewise). The UPDATE trigger carries
#   no column list — transition tables see every UPDATE statement and the
#   function filters on processed actually changing, so payload/error-only
#   updates walk an empty set and do no aggregate work.
# - _work_unit_backlog_recompute LOCKS THE BACKLOG ROW FIRST (FOR UPDATE) and
#   only then aggregates. Without that lock, READ COMMITTED lets a recompute
#   aggregate a snapshot that excludes a concurrently-committing enqueue and
#   then clobber the fast path's increment — or worse, its membership DELETE
#   blocks on the enqueue's row lock and deletes anyway after it commits,
#   because the EPQ recheck re-evaluates only the row, not the NOT EXISTS
#   subquery; a live unit would lose its backlog row permanently. Taking the
#   row lock first serializes the recompute behind the enqueue, and each
#   later statement in the function gets a fresh snapshot that sees it.
# - Trigger-side lock ordering is canonical (sorted by work_unit_key): the
#   statement triggers iterate keys sorted, and the enqueue batch is sorted
#   app-side before insert. The claim necessarily locks in scheduling order
#   (SKIP LOCKED backfill depends on it), so a rare claim-vs-enqueue lock
#   inversion can still deadlock; Postgres's detector breaks it and both
#   sides retry (the enqueue's bounded retry, the claim's next poll).
# - The reconciler/dream dedup path inserts then rolls back on the partial
#   unique index; trigger effects roll back with the transaction, so the
#   dedup loser is never counted.
# endregion

# Mirrors src/utils/work_unit.py _TASK_TYPES at this revision: a key whose
# first segment is none of these carries a tenant prefix.
TASK_TYPE_LIST = (
    "'representation','summary','dream','webhook','deletion',"
    "'reconciler','scope_backfill','scope_removal'"
)


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
    # Supports the fair claim's PARTITION BY tenant_id ORDER BY
    # (oldest_created_at, work_unit_key) window with pre-sorted input.
    if not index_exists("work_unit_backlog", "ix_work_unit_backlog_tenant_oldest_key"):
        op.create_index(
            "ix_work_unit_backlog_tenant_oldest_key",
            "work_unit_backlog",
            ["tenant_id", "oldest_created_at", "work_unit_key"],
            schema=schema,
        )
    # Serves the plain oldest-first scan and the metrics min() read.
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
                -- A tenant-less queue row means a single-tenant deployment,
                -- whose message rows all carry the bootstrap default tenant —
                -- probe the composite PK with it (an id-only predicate cannot
                -- use a (tenant_id, id) PK as a search key and would scan).
                -- The id-only probe remains as a fallback for rows that
                -- predate the default-tenant backfill.
                SELECT token_count INTO t
                FROM {schema}.messages
                WHERE tenant_id = 'default' AND id = msg_id;
                IF NOT FOUND THEN
                    SELECT token_count INTO t
                    FROM {schema}.messages WHERE id = msg_id;
                END IF;
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
        DECLARE
            upserted INTEGER;
        BEGIN
            -- Serialize against the INSERT fast path BEFORE aggregating: the
            -- lock wait ends when a concurrent enqueue commits, and the
            -- statements below then run on fresh snapshots that include it.
            -- Aggregating first would clobber that enqueue's increment, and
            -- the membership DELETE would drop a live unit's row (its EPQ
            -- recheck re-evaluates the row, not the NOT EXISTS subquery).
            PERFORM 1 FROM {schema}.work_unit_backlog
            WHERE work_unit_key = key FOR UPDATE;

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
            GET DIAGNOSTICS upserted = ROW_COUNT;
            IF upserted > 0 THEN
                -- The aggregate found pending rows; the guard below could
                -- only re-scan the same range to delete nothing.
                RETURN;
            END IF;
            -- Membership-based guard: only drop the row when the queue holds
            -- nothing unprocessed for this unit.
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
        CREATE OR REPLACE FUNCTION {schema}.work_unit_backlog_apply_insert()
        RETURNS trigger AS $$
        BEGIN
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
            RETURN NULL;
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}.work_unit_backlog_apply_update()
        RETURNS trigger AS $$
        DECLARE
            affected_key TEXT;
        BEGIN
            FOR affected_key IN
                SELECT DISTINCT n.work_unit_key
                FROM new_rows n
                JOIN old_rows o ON o.id = n.id
                WHERE o.processed IS DISTINCT FROM n.processed
                ORDER BY 1
            LOOP
                PERFORM {schema}._work_unit_backlog_recompute(affected_key);
            END LOOP;
            RETURN NULL;
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {schema}.work_unit_backlog_apply_delete()
        RETURNS trigger AS $$
        DECLARE
            affected_key TEXT;
        BEGIN
            FOR affected_key IN
                SELECT DISTINCT work_unit_key
                FROM old_rows
                WHERE NOT processed
                ORDER BY 1
            LOOP
                PERFORM {schema}._work_unit_backlog_recompute(affected_key);
            END LOOP;
            RETURN NULL;
        END $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        f"""
        DROP TRIGGER IF EXISTS trg_work_unit_backlog ON {schema}.queue;
        DROP TRIGGER IF EXISTS trg_work_unit_backlog_insert ON {schema}.queue;
        DROP TRIGGER IF EXISTS trg_work_unit_backlog_update ON {schema}.queue;
        DROP TRIGGER IF EXISTS trg_work_unit_backlog_delete ON {schema}.queue;
        CREATE TRIGGER trg_work_unit_backlog_insert
        AFTER INSERT ON {schema}.queue
        FOR EACH ROW EXECUTE FUNCTION {schema}.work_unit_backlog_apply_insert();
        CREATE TRIGGER trg_work_unit_backlog_update
        AFTER UPDATE ON {schema}.queue
        REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
        FOR EACH STATEMENT
        EXECUTE FUNCTION {schema}.work_unit_backlog_apply_update();
        CREATE TRIGGER trg_work_unit_backlog_delete
        AFTER DELETE ON {schema}.queue
        REFERENCING OLD TABLE AS old_rows
        FOR EACH STATEMENT
        EXECUTE FUNCTION {schema}.work_unit_backlog_apply_delete();
        """
    )

    # region ai
    # Backfill in two steps. Rows enqueued before this revision predate every
    # tenant_id writer, so the column is NULL even where the key carries a
    # tenant prefix — left as-is, the entire pre-upgrade backlog would share
    # the fair claim's single NULL bucket. Derive the tenant from the key
    # (first segment not a task type ⟹ tenant prefix), stamp the queue rows,
    # then aggregate. The trigger install above already holds SHARE ROW
    # EXCLUSIVE on queue, so no enqueue interleaves with either step.
    # endregion
    op.execute(
        f"""
        UPDATE {schema}.queue
        SET tenant_id = split_part(work_unit_key, ':', 1)
        WHERE tenant_id IS NULL
          AND NOT processed
          AND split_part(work_unit_key, ':', 1) NOT IN ({TASK_TYPE_LIST});
        """
    )
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
    op.execute(f"DROP TRIGGER IF EXISTS trg_work_unit_backlog_insert ON {schema}.queue")
    op.execute(f"DROP TRIGGER IF EXISTS trg_work_unit_backlog_update ON {schema}.queue")
    op.execute(f"DROP TRIGGER IF EXISTS trg_work_unit_backlog_delete ON {schema}.queue")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}.work_unit_backlog_apply_insert()")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}.work_unit_backlog_apply_update()")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}.work_unit_backlog_apply_delete()")
    op.execute(f"DROP FUNCTION IF EXISTS {schema}._work_unit_backlog_recompute(TEXT)")
    op.execute(
        f"DROP FUNCTION IF EXISTS {schema}._queue_item_token_count(TEXT, BIGINT, TEXT)"
    )
    if table_exists("work_unit_backlog"):
        op.drop_table("work_unit_backlog", schema=schema)
