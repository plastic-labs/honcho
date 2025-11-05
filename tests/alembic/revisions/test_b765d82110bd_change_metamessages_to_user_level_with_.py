"""Hooks for revision b765d82110bd (metamessages user-level migration)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

METAMESSAGE_ID = generate_nanoid()
USER_ID = generate_nanoid()
SESSION_ID = generate_nanoid()
MESSAGE_ID = generate_nanoid()
_INDEXES = (
    ("metamessages", "idx_metamessages_user_lookup"),
    ("metamessages", "idx_metamessages_session_lookup"),
    ("metamessages", "idx_metamessages_message_lookup"),
)


@register_before_upgrade("b765d82110bd")
def prepare_metamessages(verifier: MigrationVerifier) -> None:
    schema = verifier.schema
    connection = verifier.conn
    # Create app
    APP_ID = generate_nanoid()
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
            f'INSERT INTO "{schema}"."metamessages" ("public_id", "message_id", "content", "metamessage_type") '
            + "VALUES (:metamessage_id, :message_id, :content, :metamessage_type)"
        ),
        {
            "metamessage_id": METAMESSAGE_ID,
            "message_id": MESSAGE_ID,
            "content": "test-content",
            "metamessage_type": "test-type",
        },
    )


@register_after_upgrade("b765d82110bd")
def verify_metamessages_promoted(verifier: MigrationVerifier) -> None:
    """Confirm metamessages rows migrated to user-level shape."""

    verifier.assert_column_exists("metamessages", "user_id", nullable=False)
    verifier.assert_column_exists("metamessages", "session_id")
    verifier.assert_column_exists("metamessages", "message_id", nullable=True)
    verifier.assert_indexes_exist(_INDEXES)
    verifier.assert_constraint_exists(
        "metamessages", "fk_metamessages_user_id_users", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "metamessages", "fk_metamessages_session_id_sessions", "foreign_key"
    )
    verifier.assert_constraint_exists(
        "metamessages", "message_requires_session", "check"
    )
    verifier.assert_no_nulls("metamessages", "user_id")

    row = verifier.conn.execute(
        text(
            'SELECT "user_id", "session_id", "message_id", "metamessage_type", "content" '
            + f'FROM "{verifier.schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_ID},
    ).one()

    assert row.user_id == USER_ID
    assert row.session_id == SESSION_ID
    assert row.message_id == MESSAGE_ID
    assert row.metamessage_type == "test-type"
    assert row.content == "test-content"
