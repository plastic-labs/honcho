"""Hooks for revision ec8f94139b02 (codify_workspace_name_and_message_id_in_)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Test data constants
WORKSPACE_NAME_1 = "test-workspace-1"
WORKSPACE_NAME_2 = "test-workspace-2"
PEER_NAME_1 = "test-peer-1"
PEER_NAME_2 = "test-peer-2"
SESSION_NAME_1 = "test-session-1"
SESSION_NAME_2 = "test-session-2"

# Indexes that should be created by the migration
_INDEXES = (
    ("queue", "ix_queue_workspace_name"),
    ("queue", "ix_queue_message_id_not_null"),
    ("queue", "ix_queue_workspace_name_processed"),
    ("queue", "ix_queue_work_unit_key_processed_id"),
)


@register_before_upgrade("ec8f94139b02")
def prepare_codify_workspace_name_and_message_id_in(
    verifier: MigrationVerifier,
) -> None:
    """Seed state and assertions before upgrading to ec8f94139b02."""
    # Verify columns don't exist yet
    verifier.assert_column_exists("queue", "workspace_name", exists=False)
    verifier.assert_column_exists("queue", "message_id", exists=False)

    # Verify indexes don't exist yet
    verifier.assert_indexes_not_exist(_INDEXES)

    conn = verifier.conn
    schema = verifier.schema

    # Create workspaces
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "name") '
            + "VALUES (:ws_id_1, :ws_name_1), (:ws_id_2, :ws_name_2)"
        ),
        {
            "ws_id_1": generate_nanoid(),
            "ws_name_1": WORKSPACE_NAME_1,
            "ws_id_2": generate_nanoid(),
            "ws_name_2": WORKSPACE_NAME_2,
        },
    )

    # Create peers
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
            + "VALUES (:peer_id_1, :peer_name_1, :ws_name_1), "
            + "(:peer_id_2, :peer_name_2, :ws_name_2)"
        ),
        {
            "peer_id_1": generate_nanoid(),
            "peer_name_1": PEER_NAME_1,
            "ws_name_1": WORKSPACE_NAME_1,
            "peer_id_2": generate_nanoid(),
            "peer_name_2": PEER_NAME_2,
            "ws_name_2": WORKSPACE_NAME_2,
        },
    )

    # Create sessions
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name") '
            + "VALUES (:session_id_1, :session_name_1, :ws_name_1), "
            + "(:session_id_2, :session_name_2, :ws_name_2)"
        ),
        {
            "session_id_1": generate_nanoid(),
            "session_name_1": SESSION_NAME_1,
            "ws_name_1": WORKSPACE_NAME_1,
            "session_id_2": generate_nanoid(),
            "session_name_2": SESSION_NAME_2,
            "ws_name_2": WORKSPACE_NAME_2,
        },
    )

    # Create some messages to reference in queue items
    message_ids: list[str] = []
    for i in range(10):
        message_id: str = generate_nanoid()
        message_ids.append(message_id)
        workspace = WORKSPACE_NAME_1 if i % 2 == 0 else WORKSPACE_NAME_2
        session = SESSION_NAME_1 if i % 2 == 0 else SESSION_NAME_2
        peer = PEER_NAME_1 if i % 2 == 0 else PEER_NAME_2

        conn.execute(
            text(
                f'INSERT INTO "{schema}"."messages" '
                + '("public_id", "workspace_name", "session_name", "peer_name", "content", "seq_in_session") '
                + "VALUES (:msg_id, :ws_name, :session_name, :peer_name, :content, :seq)"
            ),
            {
                "msg_id": message_id,
                "ws_name": workspace,
                "session_name": session,
                "peer_name": peer,
                "content": f"test message {i}",
                "seq": i,
            },
        )

    # Get internal message IDs for queue references
    message_db_ids: list[int] = []
    for msg_id in message_ids:
        result = conn.execute(
            text(f'SELECT "id" FROM "{schema}"."messages" WHERE "public_id" = :msg_id'),
            {"msg_id": msg_id},
        ).one()
        message_db_ids.append(result.id)

    # Bulk insert 100k queue items with workspace_name and message_id in payload
    # Use efficient batch insert with generate_series
    conn.execute(text("SET LOCAL synchronous_commit = OFF"))

    # Insert queue items in three categories:
    # 1. Items with both workspace_name and message_id (60k)
    # 2. Items with workspace_name but NO message_id (30k)
    # 3. Items with workspace_name and NULL message_id value (10k)

    # Category 1: Both workspace_name and message_id (60k items)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("work_unit_key", "task_type", "payload", "processed") '
            + "SELECT "
            + "  'work-unit-' || gs::text, "
            + "  'representation', "
            + "  jsonb_build_object("
            + "    'workspace_name', CASE WHEN gs % 2 = 0 THEN :ws_name_1 ELSE :ws_name_2 END, "
            + "    'message_id', :msg_db_id_0 + (gs % 10), "
            + "    'other_field', 'value-' || gs::text"
            + "  ), "
            + "  false "
            + "FROM generate_series(1, :n) AS gs"
        ),
        {
            "ws_name_1": WORKSPACE_NAME_1,
            "ws_name_2": WORKSPACE_NAME_2,
            "msg_db_id_0": message_db_ids[0],
            "n": 60_000,
        },
    )

    # Category 2: Only workspace_name, no message_id key (30k items)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("work_unit_key", "task_type", "payload", "processed") '
            + "SELECT "
            + "  'work-unit-no-msg-' || gs::text, "
            + "  'summary', "
            + "  jsonb_build_object("
            + "    'workspace_name', CASE WHEN gs % 2 = 0 THEN :ws_name_1 ELSE :ws_name_2 END, "
            + "    'other_field', 'value-' || gs::text"
            + "  ), "
            + "  false "
            + "FROM generate_series(60001, :n) AS gs"
        ),
        {
            "ws_name_1": WORKSPACE_NAME_1,
            "ws_name_2": WORKSPACE_NAME_2,
            "n": 90_000,
        },
    )

    # Category 3: workspace_name with explicit NULL message_id (10k items)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("work_unit_key", "task_type", "payload", "processed") '
            + "SELECT "
            + "  'work-unit-null-msg-' || gs::text, "
            + "  'representation', "
            + "  jsonb_build_object("
            + "    'workspace_name', CASE WHEN gs % 2 = 0 THEN :ws_name_1 ELSE :ws_name_2 END, "
            + "    'message_id', NULL::bigint, "
            + "    'other_field', 'value-' || gs::text"
            + "  ), "
            + "  false "
            + "FROM generate_series(90001, :n) AS gs"
        ),
        {
            "ws_name_1": WORKSPACE_NAME_1,
            "ws_name_2": WORKSPACE_NAME_2,
            "n": 100_000,
        },
    )

    # Verify we have exactly 100k queue items
    count = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."queue"')).scalar()
    assert count == 100_000, f"Expected 100k queue items but found {count}"


@register_after_upgrade("ec8f94139b02")
def verify_codify_workspace_name_and_message_id_in(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of ec8f94139b02."""
    # Verify columns were added with correct nullability
    verifier.assert_column_exists("queue", "workspace_name", nullable=False)
    verifier.assert_column_exists("queue", "message_id", nullable=True)

    # Verify all indexes were created
    verifier.assert_indexes_exist(_INDEXES)

    # Verify foreign key constraint exists
    verifier.assert_constraint_exists("queue", "fk_queue_workspace_name", "foreign_key")

    conn = verifier.conn
    schema = verifier.schema

    # Verify all rows have non-null workspace_name after migration
    verifier.assert_no_nulls("queue", "workspace_name")

    # Verify data transformation: workspace_name extracted from payload
    ws1_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + 'WHERE "workspace_name" = :ws_name'
        ),
        {"ws_name": WORKSPACE_NAME_1},
    ).scalar()
    ws2_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + 'WHERE "workspace_name" = :ws_name'
        ),
        {"ws_name": WORKSPACE_NAME_2},
    ).scalar()

    # Should be roughly 50/50 split (we alternate in the insert)
    assert (
        ws1_count == 50_000
    ), f"Expected 50k items with workspace_name_1, got {ws1_count}"
    assert (
        ws2_count == 50_000
    ), f"Expected 50k items with workspace_name_2, got {ws2_count}"

    # Verify data transformation: message_id extracted from payload where it exists
    msg_id_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + 'WHERE "message_id" IS NOT NULL'
        )
    ).scalar()

    # Should be 60k items with message_id (category 1 only)
    assert (
        msg_id_count == 60_000
    ), f"Expected 60k items with message_id, got {msg_id_count}"

    # Verify items without message_id in payload have NULL in column
    null_msg_id_count = conn.execute(
        text(f'SELECT COUNT(*) FROM "{schema}"."queue" ' + 'WHERE "message_id" IS NULL')
    ).scalar()

    # Should be 40k items (30k without key + 10k with NULL value)
    assert (
        null_msg_id_count == 40_000
    ), f"Expected 40k items with NULL message_id, got {null_msg_id_count}"

    # Verify workspace_name was removed from payload
    ws_in_payload_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + "WHERE payload ? 'workspace_name'"
        )
    ).scalar()
    assert (
        ws_in_payload_count == 0
    ), f"Found {ws_in_payload_count} items still with workspace_name in payload"

    # Verify message_id was removed from payload
    msg_in_payload_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" ' + "WHERE payload ? 'message_id'"
        )
    ).scalar()
    assert (
        msg_in_payload_count == 0
    ), f"Found {msg_in_payload_count} items still with message_id in payload"

    # Verify other_field remains in payload (data preservation)
    other_field_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + "WHERE payload ? 'other_field'"
        )
    ).scalar()
    assert (
        other_field_count == 100_000
    ), f"Expected all 100k items to retain other_field in payload, got {other_field_count}"

    # Spot check: verify a specific queue item was transformed correctly
    sample_item = conn.execute(
        text(
            'SELECT "workspace_name", "message_id", "payload" '
            + f'FROM "{schema}"."queue" '
            + "WHERE work_unit_key = 'work-unit-1' "
            + "LIMIT 1"
        )
    ).one()

    assert sample_item.workspace_name == WORKSPACE_NAME_2  # gs=1 is odd, so workspace 2
    assert sample_item.message_id is not None  # Category 1 item
    payload = (
        json.loads(sample_item.payload)
        if isinstance(sample_item.payload, str)
        else sample_item.payload
    )
    assert "workspace_name" not in payload
    assert "message_id" not in payload
    assert payload.get("other_field") == "value-1"
