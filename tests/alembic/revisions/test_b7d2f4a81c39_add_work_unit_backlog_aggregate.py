"""Hooks for revision b7d2f4a81c39 (add_work_unit_backlog_aggregate)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = "backlog-workspace"
PEER_NAME = "backlog-peer"
SESSION_NAME = "backlog-session"
TENANT_ID = "default"

# One work unit per shape the backfill has to get right.
TENANT_UNIT = "representation:backfill-tenant"
TENANTLESS_UNIT = "representation:backfill-tenantless"
WEBHOOK_UNIT = "webhook:backfill"
DRAINED_UNIT = "representation:backfill-drained"

INDEXES = (
    ("work_unit_backlog", "ix_work_unit_backlog_tenant_id"),
    ("work_unit_backlog", "ix_work_unit_backlog_oldest_created_at_key"),
)


@register_before_upgrade("b7d2f4a81c39")
def prepare_add_work_unit_backlog_aggregate(verifier: MigrationVerifier) -> None:
    """Seed a queue that already holds pending work before upgrading."""
    verifier.assert_table_exists("work_unit_backlog", exists=False)

    conn = verifier.conn
    schema = verifier.schema

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "tenant_id", "name") '
            + "VALUES (:id, :tenant, :name)"
        ),
        {"id": generate_nanoid(), "tenant": TENANT_ID, "name": WORKSPACE_NAME},
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "tenant_id", "name", "workspace_name") '
            + "VALUES (:id, :tenant, :name, :ws)"
        ),
        {
            "id": generate_nanoid(),
            "tenant": TENANT_ID,
            "name": PEER_NAME,
            "ws": WORKSPACE_NAME,
        },
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "tenant_id", "name", "workspace_name") '
            + "VALUES (:id, :tenant, :name, :ws)"
        ),
        {
            "id": generate_nanoid(),
            "tenant": TENANT_ID,
            "name": SESSION_NAME,
            "ws": WORKSPACE_NAME,
        },
    )

    message_ids: list[int] = []
    for seq, token_count in enumerate((5, 7, 11, 100), start=1):
        message_id = conn.execute(
            text(
                f'INSERT INTO "{schema}"."messages" '
                + '("tenant_id", "public_id", "workspace_name", "session_name", '
                + '"peer_name", "content", "token_count", "seq_in_session") '
                + "VALUES (:tenant, :pid, :ws, :sn, :pn, :content, :tokens, :seq) "
                + "RETURNING id"
            ),
            {
                "tenant": TENANT_ID,
                "pid": generate_nanoid(),
                "ws": WORKSPACE_NAME,
                "sn": SESSION_NAME,
                "pn": PEER_NAME,
                "content": "seeded message",
                "tokens": token_count,
                "seq": seq,
            },
        ).scalar_one()
        message_ids.append(message_id)

    five_tokens, seven_tokens, eleven_tokens, hundred_tokens = message_ids

    def enqueue(
        work_unit_key: str,
        task_type: str,
        *,
        tenant_id: str | None,
        message_id: int | None,
        age_seconds: int,
        processed: bool,
    ) -> None:
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."queue" '
                + '("tenant_id", "session_id", "work_unit_key", "task_type", '
                + '"payload", "processed", "workspace_name", "message_id", "created_at") '
                + "VALUES (:tenant, NULL, :key, :task_type, '{}'::jsonb, :processed, "
                + ":ws, :message_id, now() - make_interval(secs => :age))"
            ),
            {
                "tenant": tenant_id,
                "key": work_unit_key,
                "task_type": task_type,
                "processed": processed,
                "ws": WORKSPACE_NAME,
                "message_id": message_id,
                "age": age_seconds,
            },
        )

    # A tenant-stamped representation unit: two pending items and one already done.
    enqueue(
        TENANT_UNIT,
        "representation",
        tenant_id=TENANT_ID,
        message_id=five_tokens,
        age_seconds=300,
        processed=False,
    )
    enqueue(
        TENANT_UNIT,
        "representation",
        tenant_id=TENANT_ID,
        message_id=seven_tokens,
        age_seconds=120,
        processed=False,
    )
    enqueue(
        TENANT_UNIT,
        "representation",
        tenant_id=TENANT_ID,
        message_id=hundred_tokens,
        age_seconds=600,
        processed=True,
    )
    # The unpartitioned shape: no tenant on the row, token probed by id alone.
    enqueue(
        TENANTLESS_UNIT,
        "representation",
        tenant_id=None,
        message_id=eleven_tokens,
        age_seconds=60,
        processed=False,
    )
    # Infrastructure work carries no message and therefore no tokens.
    enqueue(
        WEBHOOK_UNIT,
        "webhook",
        tenant_id=TENANT_ID,
        message_id=None,
        age_seconds=30,
        processed=False,
    )
    # Nothing pending: this unit must not be backfilled at all.
    enqueue(
        DRAINED_UNIT,
        "representation",
        tenant_id=TENANT_ID,
        message_id=hundred_tokens,
        age_seconds=900,
        processed=True,
    )


@register_after_upgrade("b7d2f4a81c39")
def verify_add_work_unit_backlog_aggregate(verifier: MigrationVerifier) -> None:
    """Assert the aggregate landed and the pending queue was backfilled exactly."""
    conn = verifier.conn
    schema = verifier.schema

    verifier.assert_table_exists("work_unit_backlog")
    verifier.assert_indexes_exist(list(INDEXES))

    def backlog_row(work_unit_key: str):
        return conn.execute(
            text(
                "SELECT tenant_id, task_type, pending_count, total_tokens, "
                + f'oldest_created_at FROM "{schema}"."work_unit_backlog" '
                + "WHERE work_unit_key = :key"
            ),
            {"key": work_unit_key},
        ).one_or_none()

    # Only units with something pending get a row.
    total_rows = conn.execute(
        text(f'SELECT COUNT(*) FROM "{schema}"."work_unit_backlog"')
    ).scalar()
    assert total_rows == 3, f"expected 3 backfilled units, found {total_rows}"
    assert backlog_row(DRAINED_UNIT) is None, "a drained unit must not be backfilled"

    tenant_unit = backlog_row(TENANT_UNIT)
    assert tenant_unit is not None
    assert tenant_unit.tenant_id == TENANT_ID
    assert tenant_unit.task_type == "representation"
    assert tenant_unit.pending_count == 2, "the processed item must not be counted"
    assert tenant_unit.total_tokens == 12, "tokens sum over the pending items only"

    oldest_pending = conn.execute(
        text(
            f'SELECT min(created_at) FROM "{schema}"."queue" '
            + "WHERE work_unit_key = :key AND NOT processed"
        ),
        {"key": TENANT_UNIT},
    ).scalar_one()
    assert tenant_unit.oldest_created_at == oldest_pending

    tenantless_unit = backlog_row(TENANTLESS_UNIT)
    assert tenantless_unit is not None
    assert tenantless_unit.tenant_id is None
    assert tenantless_unit.pending_count == 1
    assert tenantless_unit.total_tokens == 11, (
        "the token probe must work without a tenant"
    )

    webhook_unit = backlog_row(WEBHOOK_UNIT)
    assert webhook_unit is not None
    assert webhook_unit.task_type == "webhook"
    assert webhook_unit.pending_count == 1
    assert webhook_unit.total_tokens == 0, "non-representation work carries no tokens"

    # The triggers installed alongside the backfill keep maintaining the row.
    conn.execute(
        text(
            f'UPDATE "{schema}"."queue" SET processed = true '
            + "WHERE work_unit_key = :key AND NOT processed"
        ),
        {"key": WEBHOOK_UNIT},
    )
    assert backlog_row(WEBHOOK_UNIT) is None, (
        "completing a unit's last pending item must delete its backlog row"
    )
