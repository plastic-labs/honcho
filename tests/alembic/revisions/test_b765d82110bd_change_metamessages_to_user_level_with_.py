"""Hooks for revision b765d82110bd (metamessages user-level migration)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.constants import (
    MESSAGE_PUBLIC_ID,
    METAMESSAGE_PUBLIC_ID,
    SESSION_PUBLIC_ID,
    USER_ID,
)
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier

_INDEXES = (
    ("metamessages", "idx_metamessages_user_lookup"),
    ("metamessages", "idx_metamessages_session_lookup"),
    ("metamessages", "idx_metamessages_message_lookup"),
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
    verifier.assert_query_returns(
        (
            f'SELECT COUNT(*) FROM "{verifier.schema}"."metamessages" '
            'WHERE "message_id" IS NOT NULL AND "session_id" IS NULL'
        ),
        expected_value=0,
        error_message="Metamessages missing session_id after promotion",
    )

    row = verifier.conn.execute(
        text(
            'SELECT "user_id", "session_id", "message_id" '
            + f'FROM "{verifier.schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_PUBLIC_ID},
    ).one()

    assert row.user_id == USER_ID
    assert row.session_id == SESSION_PUBLIC_ID
    assert row.message_id == MESSAGE_PUBLIC_ID
