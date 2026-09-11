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

# A second tenant, whose queue rows predate every tenant_id writer: the key
# carries the tenant prefix but the column is still NULL, and the backfill has
# to derive one from the other.
PREFIXED_TENANT_ID = "tenant-prefixed-1"
PREFIXED_WORKSPACE_NAME = "prefixed-workspace"
PREFIXED_PEER_NAME = "prefixed-peer"
PREFIXED_SESSION_NAME = "prefixed-session"
PREFIXED_UNIT_TOKENS = 13

# One work unit per shape the backfill has to get right.
TENANT_UNIT = "representation:backfill-tenant"
TENANTLESS_UNIT = "representation:backfill-tenantless"
WEBHOOK_UNIT = "webhook:backfill"
DRAINED_UNIT = "representation:backfill-drained"
PREFIXED_UNIT = f"{PREFIXED_TENANT_ID}:representation:backfill-prefixed"

INDEXES = (
    ("work_unit_backlog", "ix_work_unit_backlog_tenant_oldest_key"),
    ("work_unit_backlog", "ix_work_unit_backlog_oldest_created_at_key"),
)


@register_before_upgrade("b7d2f4a81c39")
def prepare_add_work_unit_backlog_aggregate(verifier: MigrationVerifier) -> None:
    """Seed a queue that already holds pending work before upgrading."""
    verifier.assert_table_exists("work_unit_backlog", exists=False)

    conn = verifier.conn
    schema = verifier.schema

    def seed_tenant_scaffold(
        *, tenant_id: str, workspace_name: str, peer_name: str, session_name: str
    ) -> None:
        """Create the workspace, peer and session a message hangs off, under one tenant."""
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."workspaces" ("id", "tenant_id", "name") '
                + "VALUES (:id, :tenant, :name)"
            ),
            {"id": generate_nanoid(), "tenant": tenant_id, "name": workspace_name},
        )
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."peers" ("id", "tenant_id", "name", "workspace_name") '
                + "VALUES (:id, :tenant, :name, :ws)"
            ),
            {
                "id": generate_nanoid(),
                "tenant": tenant_id,
                "name": peer_name,
                "ws": workspace_name,
            },
        )
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."sessions" ("id", "tenant_id", "name", "workspace_name") '
                + "VALUES (:id, :tenant, :name, :ws)"
            ),
            {
                "id": generate_nanoid(),
                "tenant": tenant_id,
                "name": session_name,
                "ws": workspace_name,
            },
        )

    def seed_message(
        *,
        tenant_id: str,
        workspace_name: str,
        peer_name: str,
        session_name: str,
        token_count: int,
        seq_in_session: int,
    ) -> int:
        """Insert one priced message and return the id the queue references."""
        return conn.execute(
            text(
                f'INSERT INTO "{schema}"."messages" '
                + '("tenant_id", "public_id", "workspace_name", "session_name", '
                + '"peer_name", "content", "token_count", "seq_in_session") '
                + "VALUES (:tenant, :pid, :ws, :sn, :pn, :content, :tokens, :seq) "
                + "RETURNING id"
            ),
            {
                "tenant": tenant_id,
                "pid": generate_nanoid(),
                "ws": workspace_name,
                "sn": session_name,
                "pn": peer_name,
                "content": "seeded message",
                "tokens": token_count,
                "seq": seq_in_session,
            },
        ).scalar_one()

    seed_tenant_scaffold(
        tenant_id=TENANT_ID,
        workspace_name=WORKSPACE_NAME,
        peer_name=PEER_NAME,
        session_name=SESSION_NAME,
    )
    conn.execute(
        text(f'INSERT INTO "{schema}"."tenants" ("tenant_id") VALUES (:tenant)'),
        {"tenant": PREFIXED_TENANT_ID},
    )
    seed_tenant_scaffold(
        tenant_id=PREFIXED_TENANT_ID,
        workspace_name=PREFIXED_WORKSPACE_NAME,
        peer_name=PREFIXED_PEER_NAME,
        session_name=PREFIXED_SESSION_NAME,
    )

    five_tokens, seven_tokens, eleven_tokens, hundred_tokens = [
        seed_message(
            tenant_id=TENANT_ID,
            workspace_name=WORKSPACE_NAME,
            peer_name=PEER_NAME,
            session_name=SESSION_NAME,
            token_count=token_count,
            seq_in_session=seq,
        )
        for seq, token_count in enumerate((5, 7, 11, 100), start=1)
    ]
    prefixed_tenant_message = seed_message(
        tenant_id=PREFIXED_TENANT_ID,
        workspace_name=PREFIXED_WORKSPACE_NAME,
        peer_name=PREFIXED_PEER_NAME,
        session_name=PREFIXED_SESSION_NAME,
        token_count=PREFIXED_UNIT_TOKENS,
        seq_in_session=1,
    )

    def enqueue(
        work_unit_key: str,
        task_type: str,
        *,
        tenant_id: str | None,
        message_id: int | None,
        age_seconds: int,
        processed: bool,
        workspace_name: str = WORKSPACE_NAME,
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
                "ws": workspace_name,
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
    # The mid-queue upgrade shape: the key was written with a tenant prefix but
    # the tenant_id column predates every writer of it, so the backfill has to
    # derive the tenant from the key before it can aggregate — and price the
    # message through that derived tenant.
    enqueue(
        PREFIXED_UNIT,
        "representation",
        tenant_id=None,
        message_id=prefixed_tenant_message,
        age_seconds=45,
        processed=False,
        workspace_name=PREFIXED_WORKSPACE_NAME,
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
    assert total_rows == 4, f"expected 4 backfilled units, found {total_rows}"
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

    # The tenant-derivation half of the backfill: a tenant-prefixed key whose
    # column was NULL is stamped from the prefix, and only then aggregated.
    # Left underived, every such unit would share the fair claim's NULL bucket.
    stamped_tenant = conn.execute(
        text(
            f'SELECT DISTINCT tenant_id FROM "{schema}"."queue" '
            + "WHERE work_unit_key = :key"
        ),
        {"key": PREFIXED_UNIT},
    ).scalar_one()
    assert stamped_tenant == PREFIXED_TENANT_ID, (
        f"the queue row's tenant was stamped {stamped_tenant!r}, "
        + f"expected {PREFIXED_TENANT_ID!r} from the key prefix"
    )

    prefixed_unit = backlog_row(PREFIXED_UNIT)
    assert prefixed_unit is not None
    assert prefixed_unit.tenant_id == PREFIXED_TENANT_ID, (
        "the aggregate runs after the stamp, so the backlog row carries the "
        + "derived tenant rather than the NULL the queue row started with"
    )
    assert prefixed_unit.pending_count == 1
    assert prefixed_unit.total_tokens == PREFIXED_UNIT_TOKENS, (
        "the token probe is keyed on the derived tenant, so it has to resolve "
        + "the message under that tenant rather than come back empty"
    )

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
