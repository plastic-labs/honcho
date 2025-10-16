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
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("session_id", "work_unit_key", "task_type", "payload", "processed") '
            + "VALUES (NULL, :work_unit_key, :task_type, :payload, false)"
        ),
        {
            "work_unit_key": "seed-random-queue-item",
            "task_type": "representation",
            "payload": '{"marker":"seed"}',
        },
    )


@register_after_upgrade("76ffba56fe8c")
def verify_queue_created_at(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("queue", "created_at", nullable=False)
    verifier.assert_column_exists("queue", "error")
    verifier.assert_indexes_exist([("queue", "ix_queue_created_at")])

    row = verifier.conn.execute(
        text(
            f'SELECT "created_at" FROM "{verifier.schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": "seed"},
    ).one()
    assert row.created_at is not None
