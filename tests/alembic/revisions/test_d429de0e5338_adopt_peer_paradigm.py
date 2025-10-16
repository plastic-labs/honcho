"""Hooks for revision d429de0e5338 (adopt peer paradigm)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import inspect, text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

APP_ID = generate_nanoid()
APP_NAME = "app-name"
USER_ID = generate_nanoid()
USER_NAME = "user-name"
SESSION_ID = generate_nanoid()
MESSAGE_ID = generate_nanoid()
MESSAGE_CONTENT = "message-content"
COLLECTION_ID = generate_nanoid()
COLLECTION_NAME = "collection-name"
DOCUMENT_ID = generate_nanoid()
QUEUE_MARKER = "seed"


@register_before_upgrade("d429de0e5338")
def prepare_peer_paradigm(verifier: MigrationVerifier) -> None:
    """Seed legacy tables so the migration can transform real data."""

    schema = verifier.schema
    conn = verifier.conn

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."apps" ("public_id", "name") '
            + "VALUES (:app_id, :app_name)"
        ),
        {"app_id": APP_ID, "app_name": APP_NAME},
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."users" ("public_id", "name", "app_id") '
            + "VALUES (:user_id, :user_name, :app_id)"
        ),
        {"user_id": USER_ID, "user_name": USER_NAME, "app_id": APP_ID},
    )

    session_result = conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" '
            + '("public_id", "user_id", "app_id", "is_active") '
            + "VALUES (:session_id, :user_id, :app_id, true) RETURNING id"
        ),
        {"session_id": SESSION_ID, "user_id": USER_ID, "app_id": APP_ID},
    ).one()
    SESSION_DB_ID = session_result.id

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."messages" '
            + '("public_id", "session_id", "is_user", "content", "user_id", "app_id") '
            + "VALUES (:message_id, :session_id, true, :content, :user_id, :app_id)"
        ),
        {
            "message_id": MESSAGE_ID,
            "session_id": SESSION_ID,
            "content": MESSAGE_CONTENT,
            "user_id": USER_ID,
            "app_id": APP_ID,
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."collections" '
            + '("public_id", "name", "user_id", "app_id") '
            + "VALUES (:collection_id, :collection_name, :user_id, :app_id)"
        ),
        {
            "collection_id": COLLECTION_ID,
            "collection_name": COLLECTION_NAME,
            "user_id": USER_ID,
            "app_id": APP_ID,
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("public_id", "collection_id", "user_id", "app_id", "content") '
            + "VALUES (:document_id, :collection_id, :user_id, :app_id, :content)"
        ),
        {
            "document_id": DOCUMENT_ID,
            "collection_id": COLLECTION_ID,
            "user_id": USER_ID,
            "app_id": APP_ID,
            "content": "document-content",
        },
    )

    queue_payload = json.dumps({"marker": QUEUE_MARKER})
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."queue" '
            + '("session_id", "payload", "processed") '
            + "VALUES (:session_db_id, :payload, false)"
        ),
        {
            "session_db_id": SESSION_DB_ID,
            "payload": queue_payload,
        },
    )


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
    verifier.assert_column_exists("collections", "internal_metadata", nullable=False)
    verifier.assert_column_exists("documents", "peer_name", nullable=False)
    verifier.assert_column_exists("documents", "workspace_name", nullable=False)
    verifier.assert_column_exists("documents", "collection_name", nullable=False)
    verifier.assert_column_exists("documents", "internal_metadata", nullable=False)
    verifier.assert_column_exists("active_queue_sessions", "id", nullable=False)
    verifier.assert_column_exists("active_queue_sessions", "session_id", nullable=True)
    verifier.assert_column_exists("active_queue_sessions", "sender_name", nullable=True)
    verifier.assert_column_exists("active_queue_sessions", "target_name", nullable=True)
    verifier.assert_column_exists("active_queue_sessions", "task_type", nullable=False)
    verifier.assert_column_exists("messages", "session_id", exists=False)
    verifier.assert_column_exists("messages", "user_id", exists=False)
    verifier.assert_column_exists("messages", "app_id", exists=False)
    verifier.assert_column_exists("messages", "is_user", exists=False)
    verifier.assert_indexes_exist(
        [
            ("messages", "idx_messages_session_lookup"),
            ("messages", "ix_messages_peer_name"),
            ("messages", "ix_messages_workspace_name"),
        ]
    )

    conn = verifier.conn
    schema = verifier.schema

    workspace = conn.execute(
        text(
            'SELECT "id", "name", "configuration", "internal_metadata" '
            + f'FROM "{schema}"."workspaces" WHERE "id" = :workspace_id'
        ),
        {"workspace_id": APP_ID},
    ).one()
    assert workspace.name == APP_NAME
    assert workspace.configuration == {}
    assert workspace.internal_metadata == {}

    peer = conn.execute(
        text(
            'SELECT "id", "name", "workspace_name", "configuration", "internal_metadata" '
            + f'FROM "{schema}"."peers" WHERE "id" = :peer_id'
        ),
        {"peer_id": USER_ID},
    ).one()
    assert peer.name == USER_NAME
    assert peer.workspace_name == APP_NAME
    assert peer.configuration == {}
    assert peer.internal_metadata == {}

    session = conn.execute(
        text(
            'SELECT "id", "name", "workspace_name", "configuration", "internal_metadata" '
            + f'FROM "{schema}"."sessions" WHERE "id" = :session_id'
        ),
        {"session_id": SESSION_ID},
    ).one()
    assert session.name == SESSION_ID
    assert session.workspace_name == APP_NAME
    assert session.configuration == {}
    assert session.internal_metadata == {}

    session_peer = conn.execute(
        text(
            f'SELECT "peer_name" FROM "{schema}"."session_peers" '
            + 'WHERE "workspace_name" = :workspace_name '
            + 'AND "session_name" = :session_name AND "peer_name" = :peer_name'
        ),
        {
            "workspace_name": APP_NAME,
            "session_name": SESSION_ID,
            "peer_name": USER_NAME,
        },
    ).one()
    assert session_peer.peer_name == USER_NAME

    message = conn.execute(
        text(
            'SELECT "session_name", "workspace_name", "peer_name", "token_count", "internal_metadata" '
            + f'FROM "{schema}"."messages" WHERE "public_id" = :message_id'
        ),
        {"message_id": MESSAGE_ID},
    ).one()
    assert message.session_name == SESSION_ID
    assert message.workspace_name == APP_NAME
    assert message.peer_name == USER_NAME
    assert message.token_count >= 0
    assert message.internal_metadata == {}

    collection = conn.execute(
        text(
            'SELECT "id", "name", "peer_name", "workspace_name", "internal_metadata" '
            + f'FROM "{schema}"."collections" WHERE "id" = :collection_id'
        ),
        {"collection_id": COLLECTION_ID},
    ).one()
    assert collection.name == COLLECTION_NAME
    assert collection.peer_name == USER_NAME
    assert collection.workspace_name == APP_NAME
    assert collection.internal_metadata == {}

    document = conn.execute(
        text(
            'SELECT "peer_name", "workspace_name", "collection_name", "internal_metadata" '
            + f'FROM "{schema}"."documents" WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_ID},
    ).one()
    assert document.peer_name == USER_NAME
    assert document.workspace_name == APP_NAME
    assert document.collection_name == COLLECTION_NAME
    assert document.internal_metadata == {}

    queue_row = conn.execute(
        text(
            f'SELECT "session_id", "payload" FROM "{schema}"."queue" '
            + "WHERE payload->>'marker' = :marker"
        ),
        {"marker": QUEUE_MARKER},
    ).one()
    assert queue_row.session_id == SESSION_ID

    verifier.assert_constraint_exists(
        "messages", "fk_messages_session_name_sessions", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "messages", "fk_messages_peer_name_peers", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "messages", "fk_messages_workspace_name_workspaces", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "collections", "fk_collections_peer_name_peers", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "collections", "fk_collections_workspace_name_workspaces", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "peers", "fk_peers_workspace_name_workspaces", "foreign_key"
    )
    verifier.assert_constraint_exists("peers", "unique_name_workspace_peer", "unique")
    verifier.assert_constraint_exists("sessions", "unique_session_name", "unique")
    verifier.assert_constraint_exists(
        "collections", "unique_name_collection_peer", "unique"
    )
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_active_queue_session", "unique"
    )

    verifier.assert_table_exists("metamessages", exists=False)
