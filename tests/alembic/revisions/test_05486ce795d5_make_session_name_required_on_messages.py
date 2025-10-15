"""Hooks for revision 05486ce795d5 (session_name required)."""

from __future__ import annotations

from sqlalchemy import Connection, text

from migrations.utils import get_schema
from tests.alembic.constants import (
    APP_NAME,
    LEGACY_MESSAGE_PUBLIC_ID,
    LEGACY_PEER_ID,
    LEGACY_PEER_NAME,
)
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("05486ce795d5")
def seed_orphaned_message(connection: Connection) -> None:
    """Insert a message without a session_name to exercise the migration."""

    schema = get_schema()

    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
            VALUES (:peer_id, :peer_name, :workspace_name)
            ON CONFLICT (name, workspace_name) DO NOTHING
            """
        ),
        {
            "peer_id": LEGACY_PEER_ID,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": APP_NAME,
        },
    )

    connection.execute(
        text(f'DELETE FROM "{schema}"."messages" WHERE "public_id" = :public_id'),
        {"public_id": LEGACY_MESSAGE_PUBLIC_ID},
    )

    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."messages"
                ("public_id", "workspace_name", "peer_name", "session_name", "content")
            VALUES (:public_id, :workspace_name, :peer_name, NULL, :content)
            """
        ),
        {
            "public_id": LEGACY_MESSAGE_PUBLIC_ID,
            "workspace_name": APP_NAME,
            "peer_name": LEGACY_PEER_NAME,
            "content": "orphaned message",
        },
    )


@register_after_upgrade("05486ce795d5")
def verify_session_name_enforced(verifier: MigrationVerifier) -> None:
    """Ensure session_name is populated and constrained after the migration."""

    verifier.assert_column_exists("messages", "session_name", nullable=False)
    verifier.assert_column_exists("message_embeddings", "session_name", nullable=False)

    schema = verifier.schema
    conn = verifier.conn

    message = conn.execute(
        text(
            f'SELECT "session_name" FROM "{schema}"."messages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": LEGACY_MESSAGE_PUBLIC_ID},
    ).one()
    assert message.session_name == LEGACY_PEER_NAME

    session = conn.execute(
        text(
            f'SELECT "name", "workspace_name" FROM "{schema}"."sessions" '
            + 'WHERE "name" = :session_name'
        ),
        {"session_name": LEGACY_PEER_NAME},
    ).one()
    assert session.workspace_name == APP_NAME

    constraint_def = conn.execute(
        text(
            """
            SELECT pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conname = :name AND conrelid = CAST(:relid AS regclass)
            """
        ),
        {
            "name": "name_length",
            "relid": f"{schema}.collections" if schema != "public" else "collections",
        },
    ).scalar_one()
    assert "1025" in constraint_def
