"""Hooks for revision cfaff339d519 (add session last_message_at)."""

from __future__ import annotations

import datetime

import sqlalchemy as sa
from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = generate_nanoid()
PEER_NAME = generate_nanoid()
ACTIVE_SESSION_NAME = generate_nanoid()
EMPTY_SESSION_NAME = generate_nanoid()
LATEST_MESSAGE_AT = datetime.datetime(2026, 1, 3, 12, 0, tzinfo=datetime.UTC)
INDEX_NAME = "ix_sessions_workspace_last_message_at"


@register_before_upgrade("cfaff339d519")
def prepare_add_session_last_message_at(verifier: MigrationVerifier) -> None:
    """Seed sessions and messages before the activity timestamp exists."""
    verifier.assert_column_exists("sessions", "last_message_at", exists=False)
    verifier.assert_indexes_not_exist([("sessions", INDEX_NAME)])

    schema = verifier.schema
    connection = verifier.conn
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."workspaces" ("id", "name")
            VALUES (:id, :name)
            """
        ),
        {"id": generate_nanoid(), "name": WORKSPACE_NAME},
    )
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
            VALUES (:id, :name, :workspace_name)
            """
        ),
        {
            "id": generate_nanoid(),
            "name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )
    for session_name in (ACTIVE_SESSION_NAME, EMPTY_SESSION_NAME):
        connection.execute(
            text(
                f"""
                INSERT INTO "{schema}"."sessions"
                    ("id", "name", "workspace_name", "is_active")
                VALUES (:id, :name, :workspace_name, true)
                """
            ),
            {
                "id": generate_nanoid(),
                "name": session_name,
                "workspace_name": WORKSPACE_NAME,
            },
        )

    for seq, created_at in enumerate(
        (
            datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC),
            LATEST_MESSAGE_AT,
        ),
        start=1,
    ):
        connection.execute(
            text(
                f"""
                INSERT INTO "{schema}"."messages"
                    ("public_id", "session_name", "content", "token_count",
                     "seq_in_session", "created_at", "peer_name", "workspace_name")
                VALUES
                    (:public_id, :session_name, :content, 1,
                     :seq_in_session, :created_at, :peer_name, :workspace_name)
                """
            ),
            {
                "public_id": generate_nanoid(),
                "session_name": ACTIVE_SESSION_NAME,
                "content": f"message {seq}",
                "seq_in_session": seq,
                "created_at": created_at,
                "peer_name": PEER_NAME,
                "workspace_name": WORKSPACE_NAME,
            },
        )


@register_after_upgrade("cfaff339d519")
def verify_add_session_last_message_at(verifier: MigrationVerifier) -> None:
    """Verify the nullable field, historical backfill, and sorting index."""
    verifier.assert_column_exists("sessions", "last_message_at", nullable=True)
    verifier.assert_column_type("sessions", "last_message_at", sa.TIMESTAMP)
    verifier.assert_indexes_exist([("sessions", INDEX_NAME)])

    rows = verifier.conn.execute(
        text(
            f"""
            SELECT name, last_message_at
            FROM "{verifier.schema}"."sessions"
            WHERE workspace_name = :workspace_name
            """
        ),
        {"workspace_name": WORKSPACE_NAME},
    ).all()
    activity_by_session = {row.name: row.last_message_at for row in rows}

    assert activity_by_session[ACTIVE_SESSION_NAME] == LATEST_MESSAGE_AT
    assert activity_by_session[EMPTY_SESSION_NAME] is None
