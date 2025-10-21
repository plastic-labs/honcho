"""Hooks for revision 76ffba56fe8c (queue created_at/error columns)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("76ffba56fe8c")
def prepare_queue_created_at(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("queue", "created_at", nullable=False, exists=False)
    verifier.assert_column_exists("queue", "error", exists=False)
    verifier.assert_indexes_not_exist([("queue", "ix_queue_created_at")])

    conn = verifier.conn
    schema = verifier.schema

    # Bulk insert 500k additional queue items efficiently to exercise batched backfill
    conn.execute(text("SET LOCAL synchronous_commit = OFF"))
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("session_id", "work_unit_key", "task_type", "payload", "processed") '
            + "SELECT NULL, "
            + "       'seed-batch-queue-item-' || gs::text, "
            + "       'representation', "
            + "       jsonb_build_object('marker','bulk'), "
            + "       false "
            + "FROM generate_series(1, :n) AS gs"
        ),
        {"n": 500_000},
    )


@register_after_upgrade("76ffba56fe8c")
def verify_queue_created_at(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("queue", "created_at", nullable=False)
    verifier.assert_column_exists("queue", "error")
    verifier.assert_indexes_exist([("queue", "ix_queue_created_at")])

    # Ensure all rows have non-null created_at after backfill
    verifier.assert_no_nulls("queue", "created_at")
