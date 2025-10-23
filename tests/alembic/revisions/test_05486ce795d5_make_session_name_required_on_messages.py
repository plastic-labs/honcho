"""Hooks for revision 05486ce795d5 (session_name required)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = "workspace-name"
PEER_ID = generate_nanoid()
PEER_NAME = "peer-name"
SESSION_ID = generate_nanoid()
SESSION_NAME = "session-name"
MESSAGE_ID = generate_nanoid()
MESSAGE_CONTENT = "message-content"


@register_before_upgrade("05486ce795d5")
def seed_orphaned_message(verifier: MigrationVerifier) -> None:
    """Insert a message without a session_name to exercise the migration."""

    schema = verifier.schema
    connection = verifier.conn

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."workspaces" ("id", "name")
                VALUES (:workspace_id, :workspace_name)
                """
        ),
        {"workspace_id": WORKSPACE_ID, "workspace_name": WORKSPACE_NAME},
    )

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
                VALUES (:peer_id, :peer_name, :workspace_name)
                """
        ),
        {
            "peer_id": PEER_ID,
            "peer_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
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
            "public_id": MESSAGE_ID,
            "workspace_name": WORKSPACE_NAME,
            "peer_name": PEER_NAME,
            "content": MESSAGE_CONTENT,
        },
    )


@register_after_upgrade("05486ce795d5")
def verify_session_name_enforced(verifier: MigrationVerifier) -> None:
    """Ensure session_name is populated and constrained after the migration."""

    verifier.assert_column_exists("messages", "session_name", nullable=False)

    schema = verifier.schema
    conn = verifier.conn

    session = conn.execute(
        text(
            f'SELECT "name", "workspace_name" FROM "{schema}"."sessions" '
            + 'WHERE "name" = :session_name AND "workspace_name" = :workspace_name'
        ),
        {
            "session_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    ).one_or_none()
    assert (
        session is not None
    ), "Session with expected name and workspace does not exist"

    message = conn.execute(
        text(
            f'SELECT "public_id", "content" FROM "{schema}"."messages" '
            + 'WHERE "session_name" = :session_name AND "workspace_name" = :workspace_name AND "peer_name" = :peer_name'
        ),
        {
            "session_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
            "peer_name": PEER_NAME,
        },
    ).one_or_none()
    assert message is not None, "Message with expected session_name does not exist"
    assert message.content == MESSAGE_CONTENT
