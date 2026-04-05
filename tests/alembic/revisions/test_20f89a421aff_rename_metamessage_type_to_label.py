"""Hooks for revision 20f89a421aff (metamessage label rename)."""

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


@register_before_upgrade("20f89a421aff")
def prepare_metamessage_label(verifier: MigrationVerifier) -> None:
    OLD_INDEXES = (
        ("metamessages", "idx_metamessages_lookup"),
        ("metamessages", "idx_metamessages_user_lookup"),
        ("metamessages", "idx_metamessages_session_lookup"),
        ("metamessages", "idx_metamessages_message_lookup"),
    )
    verifier.assert_indexes_exist(OLD_INDEXES)
    # Check for either naming convention or plain name (backward compatibility)
    constraints = verifier.fetch_constraints("metamessages", "check")
    assert (
        "ck_metamessages_metamessage_type_length" in constraints
        or "metamessage_type_length" in constraints
    ), f"Expected constraint not found. Available: {constraints}"

    schema = verifier.schema
    connection = verifier.conn
    inspector = verifier.get_inspector()

    connection.execute(
        text(
            f'INSERT INTO "{schema}"."apps" ("public_id", "name") '
            + "VALUES (:id, :name)"
        ),
        {"id": APP_ID, "name": "verify-app"},
    )

    connection.execute(
        text(
            f'INSERT INTO "{schema}"."users" ("public_id", "name", "app_id") '
            + "VALUES (:id, :name, :app_id)"
        ),
        {"id": USER_ID, "name": "verify-user", "app_id": APP_ID},
    )

    connection.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" '
            + '("public_id", "user_id", "app_id", "is_active") '
            + "VALUES (:id, :user_id, :app_id, true)"
        ),
        {"id": SESSION_ID, "user_id": USER_ID, "app_id": APP_ID},
    )

    connection.execute(
        text(
            f'INSERT INTO "{schema}"."messages" '
            + '("public_id", "session_id", "app_id", "user_id", "is_user", "content") '
            + "VALUES (:id, :session_id, :app_id, :user_id, true, :content)"
        ),
        {
            "id": MESSAGE_ID,
            "session_id": SESSION_ID,
            "app_id": APP_ID,
            "user_id": USER_ID,
            "content": "legacy metamessage seed",
        },
    )

    connection.execute(
        text(
            f'INSERT INTO "{schema}"."metamessages" '
            + '("public_id", "metamessage_type", "content", "message_id", "user_id", "session_id", "app_id") '
            + "VALUES (:id, :type, :content, :message_id, :user_id, :session_id, :app_id)"
        ),
        {
            "id": METAMESSAGE_ID,
            "type": "seed",
            "content": "legacy metamessage",
            "message_id": MESSAGE_ID,
            "user_id": USER_ID,
            "session_id": SESSION_ID,
            "app_id": APP_ID,
        },
    )

    columns = {
        col["name"] for col in inspector.get_columns("metamessages", schema=schema)
    }
    assert "metamessage_type" in columns
    assert "label" not in columns


@register_after_upgrade("20f89a421aff")
def verify_metamessage_label(verifier: MigrationVerifier) -> None:
    NEW_INDEXES = (
        ("metamessages", "idx_metamessages_lookup"),
        ("metamessages", "idx_metamessages_user_lookup"),
        ("metamessages", "idx_metamessages_session_lookup"),
        ("metamessages", "idx_metamessages_message_lookup"),
    )
    verifier.assert_indexes_exist(NEW_INDEXES)
    verifier.assert_column_exists("metamessages", "label", nullable=False)
    verifier.assert_constraint_exists("metamessages", "label_length", "check")

    verifier.assert_constraint_exists(
        "metamessages", "metamessage_type_length", "check", exists=False
    )
    verifier.assert_column_exists("metamessages", "metamessage_type", exists=False)

    row = verifier.conn.execute(
        text(
            f'SELECT "label" FROM "{verifier.schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_ID},
    ).one()
    assert row.label == "seed"
