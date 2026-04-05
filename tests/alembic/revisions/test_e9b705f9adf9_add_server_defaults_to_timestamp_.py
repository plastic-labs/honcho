"""Hooks for revision e9b705f9adf9 (add server defaults to timestamp, boolean, and jsonb columns)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Test data IDs
WORKSPACE_ID = generate_nanoid()
PEER_ID = generate_nanoid()
SESSION_ID = generate_nanoid()
MESSAGE_ID = generate_nanoid()
COLLECTION_ID = generate_nanoid()
DOCUMENT_ID = generate_nanoid()


@register_before_upgrade("e9b705f9adf9")
def prepare_add_server_defaults(verifier: MigrationVerifier) -> None:
    """Seed state before upgrading to e9b705f9adf9.

    This migration adds server defaults to timestamp, JSONB, and boolean columns.
    We verify that columns exist but don't have server defaults before the migration.
    """
    conn = verifier.conn
    schema = verifier.schema
    inspector = verifier.get_inspector()

    # Sample timestamp columns to check - they should exist but without server defaults
    for table, column in [
        ("workspaces", "created_at"),
        ("peers", "created_at"),
        ("sessions", "created_at"),
        ("messages", "created_at"),
        ("collections", "created_at"),
        ("documents", "created_at"),
        ("queue", "created_at"),
    ]:
        columns = inspector.get_columns(table, schema=schema)
        col_info = next((c for c in columns if c["name"] == column), None)
        assert (
            col_info is not None
        ), f"Column {table}.{column} should exist before migration"

    # Create test data to ensure existing rows work after migration
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" '
            + '("id", "name", "created_at", "metadata", "internal_metadata", "configuration") '
            + "VALUES (:id, :name, NOW(), :metadata, :internal_metadata, :configuration)"
        ),
        {
            "id": WORKSPACE_ID,
            "name": "test-workspace",
            "metadata": "{}",
            "internal_metadata": "{}",
            "configuration": "{}",
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" '
            + '("id", "name", "workspace_name", "created_at", "metadata", "internal_metadata", "configuration") '
            + "VALUES (:id, :name, :workspace_name, NOW(), :metadata, :internal_metadata, :configuration)"
        ),
        {
            "id": PEER_ID,
            "name": "test-peer",
            "workspace_name": "test-workspace",
            "metadata": "{}",
            "internal_metadata": "{}",
            "configuration": "{}",
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" '
            + '("id", "name", "workspace_name", "created_at", "is_active", "metadata", "internal_metadata", "configuration") '
            + "VALUES (:id, :name, :workspace_name, NOW(), true, :metadata, :internal_metadata, :configuration)"
        ),
        {
            "id": SESSION_ID,
            "name": "test-session",
            "workspace_name": "test-workspace",
            "metadata": "{}",
            "internal_metadata": "{}",
            "configuration": "{}",
        },
    )


@register_after_upgrade("e9b705f9adf9")
def verify_add_server_defaults(verifier: MigrationVerifier) -> None:
    """Validate server defaults were added correctly to all columns."""
    conn = verifier.conn
    schema = verifier.schema
    inspector = verifier.get_inspector()

    # Verify timestamp columns have server defaults (now() function)
    timestamp_columns = [
        ("workspaces", "created_at"),
        ("peers", "created_at"),
        ("sessions", "created_at"),
        ("messages", "created_at"),
        ("message_embeddings", "created_at"),
        ("collections", "created_at"),
        ("documents", "created_at"),
        ("queue", "created_at"),
        ("webhook_endpoints", "created_at"),
        ("session_peers", "joined_at"),
        ("active_queue_sessions", "last_updated"),
    ]

    for table, column in timestamp_columns:
        columns = inspector.get_columns(table, schema=schema)
        col_info = next((c for c in columns if c["name"] == column), None)
        assert (
            col_info is not None
        ), f"Column {table}.{column} not found after migration"

        # Check that a server default exists
        default = col_info.get("default")
        assert default is not None, (
            f"Column {table}.{column} should have a server default after migration, "
            f"but default is None"
        )

    # Verify JSONB columns have server defaults (empty object '{}')
    jsonb_columns = [
        ("workspaces", "metadata"),
        ("workspaces", "internal_metadata"),
        ("workspaces", "configuration"),
        ("peers", "metadata"),
        ("peers", "internal_metadata"),
        ("peers", "configuration"),
        ("sessions", "metadata"),
        ("sessions", "internal_metadata"),
        ("sessions", "configuration"),
        ("messages", "metadata"),
        ("messages", "internal_metadata"),
        ("collections", "metadata"),
        ("collections", "internal_metadata"),
        ("documents", "internal_metadata"),
        ("session_peers", "configuration"),
        ("session_peers", "internal_metadata"),
    ]

    for table, column in jsonb_columns:
        columns = inspector.get_columns(table, schema=schema)
        col_info = next((c for c in columns if c["name"] == column), None)
        assert (
            col_info is not None
        ), f"Column {table}.{column} not found after migration"

        # Check that a server default exists
        default = col_info.get("default")
        assert default is not None, (
            f"Column {table}.{column} should have a server default after migration, "
            f"but default is None"
        )

    # Verify boolean columns have server defaults
    boolean_columns = [
        ("sessions", "is_active", "true"),
        ("queue", "processed", "false"),
    ]

    for table, column, expected_default in boolean_columns:
        columns = inspector.get_columns(table, schema=schema)
        col_info = next((c for c in columns if c["name"] == column), None)
        assert (
            col_info is not None
        ), f"Column {table}.{column} not found after migration"

        # Check that a server default exists
        default = col_info.get("default")
        assert default is not None, (
            f"Column {table}.{column} should have a server default after migration, "
            f"but default is None"
        )

        assert (
            default == expected_default
        ), f"Column {table}.{column} should have a server default of {expected_default} after migration, but default is {default}"

    # Test that defaults actually work by inserting rows without explicit values
    test_workspace_id = generate_nanoid()
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "name") '
            + "VALUES (:id, :name)"
        ),
        {"id": test_workspace_id, "name": "test-defaults-workspace"},
    )

    # Verify the inserted workspace has default values
    workspace = conn.execute(
        text(
            'SELECT "created_at", "metadata", "internal_metadata", "configuration" '
            + f'FROM "{schema}"."workspaces" WHERE "id" = :id'
        ),
        {"id": test_workspace_id},
    ).one()

    assert workspace.created_at is not None, "created_at should be auto-populated"
    assert workspace.metadata == {}, "metadata should default to empty object"
    assert (
        workspace.internal_metadata == {}
    ), "internal_metadata should default to empty object"
    assert workspace.configuration == {}, "configuration should default to empty object"

    # Test peer defaults
    test_peer_id = generate_nanoid()
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
            + "VALUES (:id, :name, :workspace_name)"
        ),
        {
            "id": test_peer_id,
            "name": "test-defaults-peer",
            "workspace_name": "test-defaults-workspace",
        },
    )

    peer = conn.execute(
        text(
            'SELECT "created_at", "metadata", "internal_metadata", "configuration" '
            + f'FROM "{schema}"."peers" WHERE "id" = :id'
        ),
        {"id": test_peer_id},
    ).one()

    assert peer.created_at is not None, "peer created_at should be auto-populated"
    assert peer.metadata == {}, "peer metadata should default to empty object"
    assert (
        peer.internal_metadata == {}
    ), "peer internal_metadata should default to empty object"
    assert peer.configuration == {}, "peer configuration should default to empty object"

    # Test session defaults (including boolean is_active)
    test_session_id = generate_nanoid()
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name") '
            + "VALUES (:id, :name, :workspace_name)"
        ),
        {
            "id": test_session_id,
            "name": "test-defaults-session",
            "workspace_name": "test-defaults-workspace",
        },
    )

    session = conn.execute(
        text(
            'SELECT "created_at", "is_active", "metadata", "internal_metadata", "configuration" '
            + f'FROM "{schema}"."sessions" WHERE "id" = :id'
        ),
        {"id": test_session_id},
    ).one()

    assert session.created_at is not None, "session created_at should be auto-populated"
    assert session.is_active is True, "session is_active should default to true"
    assert session.metadata == {}, "session metadata should default to empty object"
    assert (
        session.internal_metadata == {}
    ), "session internal_metadata should default to empty object"
    assert (
        session.configuration == {}
    ), "session configuration should default to empty object"

    # Verify pre-existing data still exists
    existing_workspace = conn.execute(
        text(f'SELECT "id" FROM "{schema}"."workspaces" WHERE "id" = :id'),
        {"id": WORKSPACE_ID},
    ).one_or_none()
    assert (
        existing_workspace is not None
    ), "Pre-existing workspace should still exist after migration"
