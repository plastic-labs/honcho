"""Hooks for revision 88b0fb10906f (webhooks and queue updates)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("88b0fb10906f")
def verify_webhooks_and_queue(verifier: MigrationVerifier) -> None:
    """Validate queue enrichment and webhook table creation."""

    verifier.assert_table_exists("webhook_endpoints")

    verifier.assert_column_exists("queue", "task_type", nullable=False)
    verifier.assert_column_exists("queue", "work_unit_key", nullable=False)
    verifier.assert_column_exists("active_queue_sessions", "work_unit_key")

    row = verifier.conn.execute(
        text(
            f'SELECT "task_type", "work_unit_key" FROM "{verifier.schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": "seed"},
    ).one()
    assert row.task_type is not None
    assert row.work_unit_key is not None and len(row.work_unit_key) > 0
