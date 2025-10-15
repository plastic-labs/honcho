"""Hooks for revision 76ffba56fe8c (queue created_at/error columns)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("76ffba56fe8c")
def verify_queue_created_at(verifier: MigrationVerifier) -> None:
    """Ensure queue rows gain created_at timestamps."""

    verifier.assert_column_exists("queue", "created_at", nullable=False)
    verifier.assert_column_exists("queue", "error")
    verifier.assert_index_exists("queue", "ix_queue_created_at")

    row = verifier.conn.execute(
        text(
            f'SELECT "created_at" FROM "{verifier.schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": "seed"},
    ).one()
    assert row.created_at is not None
