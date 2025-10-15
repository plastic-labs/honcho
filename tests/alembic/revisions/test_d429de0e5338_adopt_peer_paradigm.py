"""Hooks for revision d429de0e5338 (adopt peer paradigm)."""

from __future__ import annotations

from sqlalchemy import inspect, text

from tests.alembic.constants import (
    APP_ID,
    APP_NAME,
    COLLECTION_NAME,
    COLLECTION_PUBLIC_ID,
    DOCUMENT_PUBLIC_ID,
    MESSAGE_PUBLIC_ID,
    SESSION_PUBLIC_ID,
    USER_ID,
    USER_NAME,
)
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("d429de0e5338")
def verify_peer_paradigm(verifier: MigrationVerifier) -> None:
    """Validate the large-scale peer paradigm migration."""

    inspector = inspect(verifier.conn)

    assert inspector.has_table("workspaces", schema=verifier.schema)
    assert not inspector.has_table("apps", schema=verifier.schema)

    verifier.assert_column_exists("workspaces", "configuration", nullable=False)
    verifier.assert_column_exists("workspaces", "internal_metadata", nullable=False)
    verifier.assert_column_exists("peers", "workspace_name", nullable=False)
    verifier.assert_column_exists("peers", "configuration", nullable=False)
    verifier.assert_column_exists("sessions", "name", nullable=False)
    verifier.assert_column_exists("sessions", "workspace_name", nullable=False)
    verifier.assert_column_exists(
        "messages", "session_name", nullable=True
    )  # later changed back to required
    verifier.assert_column_exists("messages", "workspace_name", nullable=False)
    verifier.assert_column_exists("messages", "peer_name", nullable=False)
    verifier.assert_column_exists("collections", "peer_name", nullable=False)
    verifier.assert_column_exists("collections", "workspace_name", nullable=False)
    verifier.assert_column_exists("documents", "peer_name", nullable=False)
    verifier.assert_column_exists("documents", "workspace_name", nullable=False)
    verifier.assert_column_exists("documents", "collection_name", nullable=False)
    verifier.assert_column_exists("documents", "internal_metadata", nullable=False)

    message_columns = {
        col["name"] for col in inspector.get_columns("messages", schema=verifier.schema)
    }
    assert "session_id" not in message_columns

    collection_columns = {
        col["name"]
        for col in inspector.get_columns("collections", schema=verifier.schema)
    }
    assert "user_id" not in collection_columns
    assert "app_id" not in collection_columns

    document_columns = {
        col["name"]
        for col in inspector.get_columns("documents", schema=verifier.schema)
    }
    assert "collection_id" not in document_columns
    assert "user_id" not in document_columns
    assert "app_id" not in document_columns

    conn = verifier.conn
    schema = verifier.schema

    workspace = conn.execute(
        text(f'SELECT "id", "name" FROM "{schema}"."workspaces" WHERE "id" = :app_id'),
        {"app_id": APP_ID},
    ).one()
    assert workspace.id == APP_ID
    assert workspace.name == APP_NAME

    peer = conn.execute(
        text(
            f'SELECT "id", "name", "workspace_name" FROM "{schema}"."peers" '
            + 'WHERE "id" = :user_id'
        ),
        {"user_id": USER_ID},
    ).one()
    assert peer.name == USER_NAME
    assert peer.workspace_name == APP_NAME

    session = conn.execute(
        text(
            f'SELECT "id", "name", "workspace_name" FROM "{schema}"."sessions" '
            + 'WHERE "id" = :session_id'
        ),
        {"session_id": SESSION_PUBLIC_ID},
    ).one()
    assert session.name == SESSION_PUBLIC_ID
    assert session.workspace_name == APP_NAME

    message = conn.execute(
        text(
            'SELECT "session_name", "workspace_name", "peer_name" '
            + f'FROM "{schema}"."messages" WHERE "public_id" = :message_id'
        ),
        {"message_id": MESSAGE_PUBLIC_ID},
    ).one()
    assert message.session_name == SESSION_PUBLIC_ID
    assert message.workspace_name == APP_NAME
    assert message.peer_name == USER_NAME

    session_peer = conn.execute(
        text(
            f'SELECT "peer_name" FROM "{schema}"."session_peers" '
            + 'WHERE "session_name" = :session_name AND "peer_name" = :peer_name'
        ),
        {"session_name": SESSION_PUBLIC_ID, "peer_name": USER_NAME},
    ).one()
    assert session_peer.peer_name == USER_NAME

    collection = conn.execute(
        text(
            'SELECT "id", "name", "peer_name", "workspace_name" '
            + f'FROM "{schema}"."collections" WHERE "id" = :collection_id'
        ),
        {"collection_id": COLLECTION_PUBLIC_ID},
    ).one()
    assert collection.name == COLLECTION_NAME
    assert collection.peer_name == USER_NAME
    assert collection.workspace_name == APP_NAME

    document = conn.execute(
        text(
            'SELECT "id", "collection_name", "peer_name", "workspace_name" '
            + f'FROM "{schema}"."documents" WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_PUBLIC_ID},
    ).one()
    assert document.collection_name == COLLECTION_NAME
    assert document.peer_name == USER_NAME
    assert document.workspace_name == APP_NAME

    queue_row = conn.execute(
        text(
            f'SELECT "session_id" FROM "{schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": "seed"},
    ).one()
    assert queue_row.session_id == SESSION_PUBLIC_ID
