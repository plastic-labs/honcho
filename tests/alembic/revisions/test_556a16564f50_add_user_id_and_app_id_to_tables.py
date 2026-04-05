"""Hooks for revision 556a16564f50 (propagate app/user identifiers)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

APP_ID = generate_nanoid()
USER_ID = generate_nanoid()
SESSION_ID = generate_nanoid()
MESSAGE_ID = generate_nanoid()
METAMESSAGE_ID = generate_nanoid()
COLLECTION_ID = generate_nanoid()
DOCUMENT_ID = generate_nanoid()


@register_before_upgrade("556a16564f50")
def prepare_user_app_ids(verifier: MigrationVerifier) -> None:
    schema = verifier.schema
    connection = verifier.conn

    # Create app
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."apps" ("public_id", "name") '
            + "VALUES (:app_id, :name)"
        ),
        {"app_id": APP_ID, "name": "test-app"},
    )

    # Create user
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."users" ("public_id", "name", "app_id") '
            + "VALUES (:user_id, :name, :app_id)"
        ),
        {"user_id": USER_ID, "name": "test-user", "app_id": APP_ID},
    )

    # Create session
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("public_id", "user_id", "is_active") '
            + "VALUES (:session_id, :user_id, true)"
        ),
        {"session_id": SESSION_ID, "user_id": USER_ID},
    )

    # Create message
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."messages" ("public_id", "session_id", "is_user", "content") '
            + "VALUES (:message_id, :session_id, true, :content)"
        ),
        {
            "message_id": MESSAGE_ID,
            "session_id": SESSION_ID,
            "content": "test-content",
        },
    )

    # Create metamessage
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."metamessages" ("public_id", "user_id", "content", "metamessage_type") '
            + "VALUES (:metamessage_id, :user_id, :content, :metamessage_type)"
        ),
        {
            "metamessage_id": METAMESSAGE_ID,
            "user_id": USER_ID,
            "content": "test-content",
            "metamessage_type": "test-type",
        },
    )

    # Create collection
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."collections" ("public_id", "name", "user_id") '
            + "VALUES (:collection_id, :name, :user_id)"
        ),
        {
            "collection_id": COLLECTION_ID,
            "name": "test-collection",
            "user_id": USER_ID,
        },
    )

    # Create document
    connection.execute(
        text(
            f'INSERT INTO "{schema}"."documents" ("public_id", "content", "collection_id") '
            + "VALUES (:document_id, :content, :collection_id)"
        ),
        {
            "document_id": DOCUMENT_ID,
            "content": "test-document-content",
            "collection_id": COLLECTION_ID,
        },
    )


@register_after_upgrade("556a16564f50")
def verify_app_and_user_ids(verifier: MigrationVerifier) -> None:
    """Ensure app/user identifiers are present after the upgrade."""

    verifier.assert_column_exists("sessions", "app_id", nullable=False)
    verifier.assert_indexes_exist([("sessions", "ix_sessions_app_id")])

    verifier.assert_column_exists("messages", "app_id", nullable=False)
    verifier.assert_column_exists("messages", "user_id", nullable=False)
    verifier.assert_indexes_exist(
        [
            ("messages", "ix_messages_app_id"),
            ("messages", "ix_messages_user_id"),
        ]
    )

    verifier.assert_column_exists("metamessages", "app_id", nullable=False)
    verifier.assert_indexes_exist([("metamessages", "ix_metamessages_app_id")])

    verifier.assert_column_exists("collections", "app_id", nullable=False)
    verifier.assert_indexes_exist([("collections", "ix_collections_app_id")])

    verifier.assert_column_exists("documents", "app_id", nullable=False)
    verifier.assert_column_exists("documents", "user_id", nullable=False)
    verifier.assert_indexes_exist(
        [
            ("documents", "ix_documents_app_id"),
            ("documents", "ix_documents_user_id"),
        ]
    )

    conn = verifier.conn
    schema = verifier.schema

    session_row = conn.execute(
        text(
            f'SELECT "app_id" FROM "{schema}"."sessions" '
            + 'WHERE "public_id" = :session_public_id'
        ),
        {"session_public_id": SESSION_ID},
    ).one()
    assert session_row.app_id == APP_ID

    message_row = conn.execute(
        text(
            f'SELECT "app_id", "user_id" FROM "{schema}"."messages" '
            + 'WHERE "public_id" = :message_public_id'
        ),
        {"message_public_id": MESSAGE_ID},
    ).one()
    assert message_row.app_id == APP_ID
    assert message_row.user_id == USER_ID

    metamessage_row = conn.execute(
        text(
            f'SELECT "app_id" FROM "{schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_ID},
    ).one()
    assert metamessage_row.app_id == APP_ID

    collection_row = conn.execute(
        text(
            f'SELECT "app_id" FROM "{schema}"."collections" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": COLLECTION_ID},
    ).one()
    assert collection_row.app_id == APP_ID

    document_row = conn.execute(
        text(
            f'SELECT "app_id", "user_id" FROM "{schema}"."documents" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": DOCUMENT_ID},
    ).one()
    assert document_row.app_id == APP_ID
    assert document_row.user_id == USER_ID
