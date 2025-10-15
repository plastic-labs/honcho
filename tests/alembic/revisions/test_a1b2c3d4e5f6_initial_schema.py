"""Hooks for initial schema revision a1b2c3d4e5f6."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.constants import (
    APP_ID,
    APP_NAME,
    COLLECTION_NAME,
    COLLECTION_PUBLIC_ID,
    DOCUMENT_PUBLIC_ID,
    MESSAGE_PUBLIC_ID,
    METAMESSAGE_PUBLIC_ID,
    SESSION_PUBLIC_ID,
    USER_ID,
    USER_NAME,
)
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("a1b2c3d4e5f6")
def seed_initial_dataset(verifier: MigrationVerifier) -> None:
    """Populate baseline rows required for downstream migration checks."""

    conn = verifier.conn
    schema = verifier.schema

    conn.execute(
        text(f'DELETE FROM "{schema}"."metamessages" WHERE "public_id" = :public_id'),
        {"public_id": METAMESSAGE_PUBLIC_ID},
    )
    conn.execute(
        text(f'DELETE FROM "{schema}"."messages" WHERE "public_id" = :public_id'),
        {"public_id": MESSAGE_PUBLIC_ID},
    )
    conn.execute(
        text(f'DELETE FROM "{schema}"."sessions" WHERE "public_id" = :public_id'),
        {"public_id": SESSION_PUBLIC_ID},
    )
    conn.execute(
        text(f'DELETE FROM "{schema}"."users" WHERE "public_id" = :public_id'),
        {"public_id": USER_ID},
    )
    conn.execute(
        text(f'DELETE FROM "{schema}"."apps" WHERE "public_id" = :public_id'),
        {"public_id": APP_ID},
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."apps" ("public_id", "name") '
            + "VALUES (:public_id, :name)"
        ),
        {"public_id": APP_ID, "name": APP_NAME},
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."users" '
            + '("public_id", "name", "app_id") '
            + "VALUES (:public_id, :name, :app_id)"
        ),
        {
            "public_id": USER_ID,
            "name": USER_NAME,
            "app_id": APP_ID,
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" '
            + '("public_id", "user_id", "is_active") '
            + "VALUES (:public_id, :user_id, :is_active)"
        ),
        {
            "public_id": SESSION_PUBLIC_ID,
            "user_id": USER_ID,
            "is_active": True,
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."messages" '
            + '("public_id", "session_id", "is_user", "content") '
            + "VALUES (:public_id, :session_id, :is_user, :content)"
        ),
        {
            "public_id": MESSAGE_PUBLIC_ID,
            "session_id": SESSION_PUBLIC_ID,
            "is_user": True,
            "content": "seed message",
        },
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."metamessages" '
            + '("public_id", "metamessage_type", "content", "message_id") '
            + "VALUES (:public_id, :metamessage_type, :content, :message_id)"
        ),
        {
            "public_id": METAMESSAGE_PUBLIC_ID,
            "metamessage_type": "seed",
            "content": "seed metamessage",
            "message_id": MESSAGE_PUBLIC_ID,
        },
    )

    conn.execute(
        text(f'DELETE FROM "{schema}"."collections" WHERE "public_id" = :public_id'),
        {"public_id": COLLECTION_PUBLIC_ID},
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."collections" '
            + '("public_id", "name", "user_id") '
            + "VALUES (:public_id, :name, :user_id)"
        ),
        {
            "public_id": COLLECTION_PUBLIC_ID,
            "name": COLLECTION_NAME,
            "user_id": USER_ID,
        },
    )

    conn.execute(
        text(f'DELETE FROM "{schema}"."documents" WHERE "public_id" = :public_id'),
        {"public_id": DOCUMENT_PUBLIC_ID},
    )

    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("public_id", "collection_id", "content") '
            + "VALUES (:public_id, :collection_id, :content)"
        ),
        {
            "public_id": DOCUMENT_PUBLIC_ID,
            "collection_id": COLLECTION_PUBLIC_ID,
            "content": "Seed document",
        },
    )

    # Seed queue item with identifiable payload marker
    conn.execute(
        text(f'DELETE FROM "{schema}"."queue" WHERE payload->>\'marker\' = :marker'),
        {"marker": "seed"},
    )

    conn.execute(
        text(
            f"""
            INSERT INTO "{schema}"."queue" ("session_id", "payload", "processed")
            VALUES (
                (SELECT id FROM "{schema}"."sessions" WHERE public_id = :session_public_id),
                CAST(:payload AS jsonb),
                false
            )
            """
        ),
        {
            "payload": '{"task_type": "representation", "marker": "seed"}',
            "session_public_id": SESSION_PUBLIC_ID,
        },
    )
