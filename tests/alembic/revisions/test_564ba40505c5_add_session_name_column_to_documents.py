"""Hooks for revision 564ba40505c5 (documents session_name column)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = generate_nanoid()
PEER_ID = generate_nanoid()
PEER_NAME = generate_nanoid()
SESSION_ID = generate_nanoid()
SESSION_NAME = generate_nanoid()
COLLECTION_ID = generate_nanoid()
COLLECTION_NAME = generate_nanoid()
DOCUMENT_ID = generate_nanoid()


@register_before_upgrade("564ba40505c5")
def seed_document_session_metadata(verifier: MigrationVerifier) -> None:
    schema = verifier.schema
    connection = verifier.conn
    verifier.assert_column_exists("documents", "session_name", exists=False)
    verifier.assert_indexes_not_exist([("documents", "idx_documents_session_name")])
    verifier.assert_constraint_exists(
        "documents", "fk_documents_session_workspace", "foreign_key", exists=False
    )

    # Seed workspaces, peers, and collections
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
            INSERT INTO "{schema}"."sessions"
                ("id", "name", "workspace_name", "is_active")
            VALUES (:session_id, :session_name, :workspace_name, true)
            """
        ),
        {
            "session_id": SESSION_ID,
            "session_name": SESSION_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Create session_peers entry so downgrade can find user_id
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."session_peers"
                ("workspace_name", "session_name", "peer_name")
            VALUES (:workspace_name, :session_name, :peer_name)
            """
        ),
        {
            "workspace_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
            "peer_name": PEER_NAME,
        },
    )

    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."collections"
                ("id", "name", "peer_name", "workspace_name")
            VALUES (:collection_id, :collection_name, :peer_name, :workspace_name)
            """
        ),
        {
            "collection_id": COLLECTION_ID,
            "collection_name": COLLECTION_NAME,
            "peer_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Add document with session_name in internal_metadata
    internal_metadata = json.dumps(
        {
            "session_name": SESSION_NAME,
        }
    )

    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."documents"
                (
                    "id",
                    "collection_name",
                    "peer_name",
                    "workspace_name",
                    "content",
                    "internal_metadata"
                )
            VALUES (
                :document_id,
                :collection_name,
                :peer_name,
                :workspace_name,
                :content,
                :internal_metadata
            )
            """
        ),
        {
            "document_id": DOCUMENT_ID,
            "collection_name": COLLECTION_NAME,
            "peer_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
            "content": "seed content",
            "internal_metadata": internal_metadata,
        },
    )


@register_after_upgrade("564ba40505c5")
def verify_document_session_column(verifier: MigrationVerifier) -> None:
    """Ensure the session_name column reflects migrated metadata."""

    verifier.assert_column_exists("documents", "session_name", nullable=True)
    verifier.assert_indexes_exist([("documents", "idx_documents_session_name")])
    verifier.assert_constraint_exists(
        "documents", "fk_documents_session_workspace", "foreign_key"
    )

    row = verifier.conn.execute(
        text(
            f'SELECT "session_name" FROM "{verifier.schema}"."documents" '
            + 'WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_ID},
    ).one()
    assert row.session_name == SESSION_NAME
