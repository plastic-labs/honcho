"""Hooks for revision 08894082221a (observer/observed refactor)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import inspect, text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

COLLECTION_ID = generate_nanoid()
COLLECTION_NAME = "global_representation"
DOCUMENT_ID = generate_nanoid()
DOCUMENT_NAME = "document-name"
LEGACY_PEER_ID = generate_nanoid()
LEGACY_PEER_NAME = "legacy-peer"
WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = "workspace-name"


# Based on migration logic: when name='global_representation', observer=peer_name and observed=peer_name
EXPECTED_OBSERVED = LEGACY_PEER_NAME


@register_before_upgrade("08894082221a")
def prepare_observer_observed(verifier: MigrationVerifier) -> None:
    schema = verifier.schema
    conn = verifier.conn

    # Create workspace
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "name") '
            + "VALUES (:workspace_id, :workspace_name)"
        ),
        {"workspace_id": WORKSPACE_ID, "workspace_name": WORKSPACE_NAME},
    )

    # Create peer
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
            + "VALUES (:peer_id, :peer_name, :workspace_name)"
        ),
        {
            "peer_id": LEGACY_PEER_ID,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Create session with session_peers relationship
    session_id = generate_nanoid()
    session_name = "test-session"
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name", "is_active") '
            + "VALUES (:session_id, :session_name, :workspace_name, true)"
        ),
        {
            "session_id": session_id,
            "session_name": session_name,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Create session_peers entry so downgrade can find user_id
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."session_peers" ("workspace_name", "session_name", "peer_name") '
            + "VALUES (:workspace_name, :session_name, :peer_name)"
        ),
        {
            "workspace_name": WORKSPACE_NAME,
            "session_name": session_name,
            "peer_name": LEGACY_PEER_NAME,
        },
    )

    # Create collection with the old schema (has 'name' field)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."collections" '
            + '("id", "name", "peer_name", "workspace_name") '
            + "VALUES (:collection_id, :collection_name, :peer_name, :workspace_name)"
        ),
        {
            "collection_id": COLLECTION_ID,
            "collection_name": COLLECTION_NAME,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Create document with the old schema (has 'collection_name' field)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "collection_name", "peer_name", "workspace_name", "content", "session_name") '
            + "VALUES (:document_id, :collection_name, :peer_name, :workspace_name, :content, :session_name)"
        ),
        {
            "document_id": DOCUMENT_ID,
            "collection_name": COLLECTION_NAME,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
            "content": "test content",
            "session_name": session_name,
        },
    )


@register_after_upgrade("08894082221a")
def verify_observer_observed_migration(verifier: MigrationVerifier) -> None:
    """Assert that collections/documents now rely on observer/observed fields."""

    inspector = inspect(verifier.conn)

    collection_columns = {
        col["name"]
        for col in inspector.get_columns("collections", schema=verifier.schema)
    }
    assert "observer" in collection_columns
    assert "observed" in collection_columns
    assert "name" not in collection_columns

    document_columns = {
        col["name"]
        for col in inspector.get_columns("documents", schema=verifier.schema)
    }
    assert "observer" in document_columns
    assert "observed" in document_columns
    assert "collection_name" not in document_columns

    verifier.assert_indexes_exist(
        [
            ("collections", "idx_collections_observer"),
            ("collections", "idx_collections_observed"),
            ("documents", "idx_documents_observer"),
            ("documents", "idx_documents_observed"),
        ]
    )
    verifier.assert_constraint_exists(
        "collections", "unique_observer_observed_collection", "unique"
    )
    verifier.assert_constraint_exists(
        "documents", "documents_observer_observed_workspace_name_fkey", "foreign_key"
    )

    collection = verifier.conn.execute(
        text(
            'SELECT "observer", "observed", "workspace_name" '
            + f'FROM "{verifier.schema}"."collections" '
            + 'WHERE "id" = :collection_id'
        ),
        {"collection_id": COLLECTION_ID},
    ).one()
    assert collection.observer == LEGACY_PEER_NAME
    assert collection.observed == EXPECTED_OBSERVED
    assert collection.workspace_name == WORKSPACE_NAME

    document = verifier.conn.execute(
        text(
            'SELECT "observer", "observed", "workspace_name" '
            + f'FROM "{verifier.schema}"."documents" '
            + 'WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_ID},
    ).one()
    assert document.observer == LEGACY_PEER_NAME
    assert document.observed == EXPECTED_OBSERVED
    assert document.workspace_name == WORKSPACE_NAME
