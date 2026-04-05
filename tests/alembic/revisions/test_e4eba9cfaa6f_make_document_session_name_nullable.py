"""Hooks for revision e4eba9cfaa6f (make_document_session_name_nullable)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Test IDs for seeding data
WORKSPACE_NAME = generate_nanoid()
PEER_NAME = generate_nanoid()
SESSION_NAME = generate_nanoid()
DOCUMENT_ID = generate_nanoid()


@register_before_upgrade("e4eba9cfaa6f")
def prepare_make_document_session_name_nullable(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to e4eba9cfaa6f."""
    # Verify session_name column is NOT nullable before migration
    verifier.assert_column_exists("documents", "session_name", nullable=False)

    schema = verifier.schema
    connection = verifier.conn

    # Seed workspace
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."workspaces" ("id", "name")
            VALUES (:id, :name)
            """
        ),
        {"id": generate_nanoid(), "name": WORKSPACE_NAME},
    )

    # Seed peer
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
            VALUES (:id, :name, :workspace_name)
            """
        ),
        {"id": generate_nanoid(), "name": PEER_NAME, "workspace_name": WORKSPACE_NAME},
    )

    # Seed session
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name", "is_active")
            VALUES (:id, :name, :workspace_name, true)
            """
        ),
        {
            "id": generate_nanoid(),
            "name": SESSION_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Seed collection
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."collections"
                ("id", "workspace_name", "observer", "observed")
            VALUES (:id, :workspace_name, :observer, :observed)
            """
        ),
        {
            "id": generate_nanoid(),
            "workspace_name": WORKSPACE_NAME,
            "observer": PEER_NAME,
            "observed": PEER_NAME,
        },
    )

    # Seed document with session_name (required before migration)
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."documents"
                ("id", "workspace_name", "observer", "observed", "content", "session_name")
            VALUES (:id, :workspace_name, :observer, :observed, :content, :session_name)
            """
        ),
        {
            "id": DOCUMENT_ID,
            "workspace_name": WORKSPACE_NAME,
            "observer": PEER_NAME,
            "observed": PEER_NAME,
            "content": "Test document with session",
            "session_name": SESSION_NAME,
        },
    )


@register_after_upgrade("e4eba9cfaa6f")
def verify_make_document_session_name_nullable(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of e4eba9cfaa6f."""
    schema = verifier.schema
    connection = verifier.conn

    # Verify session_name column IS nullable after migration
    verifier.assert_column_exists("documents", "session_name", nullable=True)

    # Verify existing document still has its session_name intact
    row = connection.execute(
        text(f'SELECT "session_name" FROM "{schema}"."documents" WHERE "id" = :id'),
        {"id": DOCUMENT_ID},
    ).one()
    assert row.session_name == SESSION_NAME

    # Verify we can now insert a document without session_name (NULL)
    null_session_doc_id = generate_nanoid()
    connection.execute(
        text(
            f"""
            INSERT INTO "{schema}"."documents"
                ("id", "workspace_name", "observer", "observed", "content", "session_name")
            VALUES (:id, :workspace_name, :observer, :observed, :content, NULL)
            """
        ),
        {
            "id": null_session_doc_id,
            "workspace_name": WORKSPACE_NAME,
            "observer": PEER_NAME,
            "observed": PEER_NAME,
            "content": "Test document without session (global)",
        },
    )

    # Verify the NULL was persisted
    null_row = connection.execute(
        text(f'SELECT "session_name" FROM "{schema}"."documents" WHERE "id" = :id'),
        {"id": null_session_doc_id},
    ).one()
    assert null_row.session_name is None
