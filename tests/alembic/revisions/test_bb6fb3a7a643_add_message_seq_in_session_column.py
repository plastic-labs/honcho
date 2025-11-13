"""Hooks for revision bb6fb3a7a643 (add_message_seq_in_session_column)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import BigInteger, text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = "workspace-seq-test"
PEER_ID = generate_nanoid()
PEER_NAME = "peer-seq-test"
SESSION_1_ID = generate_nanoid()
SESSION_1_NAME = "session-1"
SESSION_2_ID = generate_nanoid()
SESSION_2_NAME = "session-2"


@register_before_upgrade("bb6fb3a7a643")
def prepare_add_message_seq_in_session_column(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to bb6fb3a7a643."""
    schema = verifier.schema
    connection = verifier.conn

    # Assert column doesn't exist yet
    verifier.assert_column_exists("messages", "seq_in_session", exists=False)

    # Insert workspace
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."workspaces" ("id", "name")
            VALUES (:workspace_id, :workspace_name)
            """
        ),
        {"workspace_id": WORKSPACE_ID, "workspace_name": WORKSPACE_NAME},
    )

    # Insert peer
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

    # Insert two sessions
    for session_id, session_name in [
        (SESSION_1_ID, SESSION_1_NAME),
        (SESSION_2_ID, SESSION_2_NAME),
    ]:
        connection.execute(
            text(
                f"""
                INSERT INTO "{schema}"."sessions"
                    ("id", "name", "workspace_name")
                VALUES (:session_id, :session_name, :workspace_name)
                """
            ),
            {
                "session_id": session_id,
                "session_name": session_name,
                "workspace_name": WORKSPACE_NAME,
            },
        )

    # Insert multiple messages in each session to test sequence numbering
    # Session 1: 5 messages
    for i in range(5):
        connection.execute(
            text(
                f"""
                INSERT INTO "{schema}"."messages"
                    ("public_id", "workspace_name", "peer_name", "session_name", "content")
                VALUES (:public_id, :workspace_name, :peer_name, :session_name, :content)
                """
            ),
            {
                "public_id": generate_nanoid(),
                "workspace_name": WORKSPACE_NAME,
                "peer_name": PEER_NAME,
                "session_name": SESSION_1_NAME,
                "content": f"Message {i + 1} in session 1",
            },
        )

    # Session 2: 3 messages
    for i in range(3):
        connection.execute(
            text(
                f"""
                INSERT INTO "{schema}"."messages"
                    ("public_id", "workspace_name", "peer_name", "session_name", "content")
                VALUES (:public_id, :workspace_name, :peer_name, :session_name, :content)
                """
            ),
            {
                "public_id": generate_nanoid(),
                "workspace_name": WORKSPACE_NAME,
                "peer_name": PEER_NAME,
                "session_name": SESSION_2_NAME,
                "content": f"Message {i + 1} in session 2",
            },
        )


@register_after_upgrade("bb6fb3a7a643")
def verify_add_message_seq_in_session_column(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of bb6fb3a7a643."""
    schema = verifier.schema
    conn = verifier.conn

    # Assert column exists and is not nullable
    verifier.assert_column_exists("messages", "seq_in_session", nullable=False)

    # Assert column type is BigInteger
    verifier.assert_column_type("messages", "seq_in_session", BigInteger)

    # Assert no null values in the column
    verifier.assert_no_nulls("messages", "seq_in_session")

    # Assert unique constraint exists
    verifier.assert_constraint_exists(
        "messages", "uq_messages_session_seq", "unique", exists=True
    )

    # Verify sequences start at 1 and are contiguous for session 1
    # Order by id to maintain insertion order, not by seq_in_session
    session_1_messages = conn.execute(
        text(
            f"""
            SELECT "content", "seq_in_session"
            FROM "{schema}"."messages"
            WHERE "workspace_name" = :workspace_name
              AND "session_name" = :session_name
            ORDER BY "id"
            """
        ),
        {
            "workspace_name": WORKSPACE_NAME,
            "session_name": SESSION_1_NAME,
        },
    ).fetchall()

    assert len(session_1_messages) == 5, "Expected 5 messages in session 1"
    for i, (content, seq) in enumerate(session_1_messages, start=1):
        assert (
            seq == i
        ), f"Expected seq_in_session={i} for message '{content}', got {seq}"

    # Verify sequences start at 1 and are contiguous for session 2
    session_2_messages = conn.execute(
        text(
            f"""
            SELECT "content", "seq_in_session"
            FROM "{schema}"."messages"
            WHERE "workspace_name" = :workspace_name
              AND "session_name" = :session_name
            ORDER BY "id"
            """
        ),
        {
            "workspace_name": WORKSPACE_NAME,
            "session_name": SESSION_2_NAME,
        },
    ).fetchall()

    assert len(session_2_messages) == 3, "Expected 3 messages in session 2"
    for i, (content, seq) in enumerate(session_2_messages, start=1):
        assert (
            seq == i
        ), f"Expected seq_in_session={i} for message '{content}', got {seq}"

    # Verify uniqueness constraint: no duplicate (workspace, session, seq) tuples
    duplicate_check = conn.execute(
        text(
            f"""
            SELECT "workspace_name", "session_name", "seq_in_session", COUNT(*)
            FROM "{schema}"."messages"
            WHERE "workspace_name" = :workspace_name
            GROUP BY "workspace_name", "session_name", "seq_in_session"
            HAVING COUNT(*) > 1
            """
        ),
        {"workspace_name": WORKSPACE_NAME},
    ).fetchall()

    assert (
        len(duplicate_check) == 0
    ), f"Found duplicate seq_in_session values: {duplicate_check}"
