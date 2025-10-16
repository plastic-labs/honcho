"""Hooks for revision 88b0fb10906f (webhooks and queue updates)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

PAYLOAD_MARKER = generate_nanoid()
WORKSPACE_NAME = generate_nanoid()
SESSION_NAME = generate_nanoid()
SENDER_NAME = generate_nanoid()
TARGET_NAME = generate_nanoid()


@register_before_upgrade("88b0fb10906f")
def prepare_webhooks(verifier: MigrationVerifier) -> None:
    # Validate objects introduced by the migration are absent
    verifier.assert_table_exists("webhook_endpoints", exists=False)
    verifier.assert_column_exists("queue", "task_type", exists=False)
    verifier.assert_column_exists("queue", "work_unit_key", exists=False)

    verifier.assert_column_exists(
        "active_queue_sessions", "work_unit_key", exists=False
    )
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique", exists=False
    )

    # Validate objects removed by the migration still exist
    verifier.assert_column_exists("active_queue_sessions", "session_id")
    verifier.assert_column_exists("active_queue_sessions", "sender_name")
    verifier.assert_column_exists("active_queue_sessions", "target_name")
    verifier.assert_column_exists("active_queue_sessions", "task_type")
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_active_queue_session", "unique"
    )

    payload = json.dumps(
        {
            "marker": PAYLOAD_MARKER,
            "task_type": "representation",
            "workspace_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
            "sender_name": SENDER_NAME,
            "target_name": TARGET_NAME,
        }
    )

    verifier.conn.execute(
        text(f'INSERT INTO "{verifier.schema}"."queue" ("payload") VALUES (:payload)'),
        {"payload": payload},
    )


@register_after_upgrade("88b0fb10906f")
def verify_webhooks_and_queue(verifier: MigrationVerifier) -> None:
    """Validate queue enrichment and webhook table creation."""

    verifier.assert_table_exists("webhook_endpoints")

    verifier.assert_column_exists("queue", "task_type", nullable=False)
    verifier.assert_column_exists("queue", "work_unit_key", nullable=False)
    verifier.assert_column_exists("active_queue_sessions", "work_unit_key")
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique"
    )

    verifier.assert_column_exists("active_queue_sessions", "session_id", exists=False)
    verifier.assert_column_exists("active_queue_sessions", "sender_name", exists=False)
    verifier.assert_column_exists("active_queue_sessions", "target_name", exists=False)
    verifier.assert_column_exists("active_queue_sessions", "task_type", exists=False)
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_active_queue_session", "unique", exists=False
    )

    row = verifier.conn.execute(
        text(
            f'SELECT "task_type", "work_unit_key" FROM "{verifier.schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": PAYLOAD_MARKER},
    ).one()

    assert row.task_type == "representation"
    expected_key = (
        f"representation:{WORKSPACE_NAME}:{SESSION_NAME}:{SENDER_NAME}:{TARGET_NAME}"
    )
    assert row.work_unit_key == expected_key
