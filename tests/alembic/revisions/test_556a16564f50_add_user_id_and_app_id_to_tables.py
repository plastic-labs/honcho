"""Hooks for revision 556a16564f50 (propagate app/user identifiers)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.constants import (
    APP_ID,
    COLLECTION_PUBLIC_ID,
    DOCUMENT_PUBLIC_ID,
    MESSAGE_PUBLIC_ID,
    METAMESSAGE_PUBLIC_ID,
    SESSION_PUBLIC_ID,
    USER_ID,
)
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("556a16564f50")
def verify_app_and_user_ids(verifier: MigrationVerifier) -> None:
    """Ensure app/user identifiers are present after the upgrade."""

    verifier.assert_column_exists("sessions", "app_id", nullable=False)
    verifier.assert_index_exists("sessions", "ix_sessions_app_id")

    verifier.assert_column_exists("messages", "app_id", nullable=False)
    verifier.assert_column_exists("messages", "user_id", nullable=False)
    verifier.assert_indexes_exist(
        [
            ("messages", "ix_messages_app_id"),
            ("messages", "ix_messages_user_id"),
        ]
    )

    verifier.assert_column_exists("metamessages", "app_id", nullable=False)
    verifier.assert_index_exists("metamessages", "ix_metamessages_app_id")

    verifier.assert_column_exists("collections", "app_id", nullable=False)
    verifier.assert_index_exists("collections", "ix_collections_app_id")

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
        {"session_public_id": SESSION_PUBLIC_ID},
    ).one()
    assert session_row.app_id == APP_ID

    message_row = conn.execute(
        text(
            f'SELECT "app_id", "user_id" FROM "{schema}"."messages" '
            + 'WHERE "public_id" = :message_public_id'
        ),
        {"message_public_id": MESSAGE_PUBLIC_ID},
    ).one()
    assert message_row.app_id == APP_ID
    assert message_row.user_id == USER_ID

    metamessage_row = conn.execute(
        text(
            f'SELECT "app_id" FROM "{schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_PUBLIC_ID},
    ).one()
    assert metamessage_row.app_id == APP_ID

    collection_row = conn.execute(
        text(
            f'SELECT "app_id" FROM "{schema}"."collections" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": COLLECTION_PUBLIC_ID},
    ).one()
    assert collection_row.app_id == APP_ID

    document_row = conn.execute(
        text(
            f'SELECT "app_id", "user_id" FROM "{schema}"."documents" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": DOCUMENT_PUBLIC_ID},
    ).one()
    assert document_row.app_id == APP_ID
    assert document_row.user_id == USER_ID
