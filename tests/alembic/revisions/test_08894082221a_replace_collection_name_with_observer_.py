"""Hooks for revision 08894082221a (observer/observed refactor)."""

from __future__ import annotations

from sqlalchemy import inspect, text

from tests.alembic.constants import (
    APP_NAME,
    COLLECTION_PUBLIC_ID,
    DOCUMENT_PUBLIC_ID,
    USER_NAME,
)
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


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
        {"collection_id": COLLECTION_PUBLIC_ID},
    ).one()
    assert collection.observer == USER_NAME
    assert collection.observed == USER_NAME
    assert collection.workspace_name == APP_NAME

    document = verifier.conn.execute(
        text(
            'SELECT "observer", "observed", "workspace_name" '
            + f'FROM "{verifier.schema}"."documents" '
            + 'WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_PUBLIC_ID},
    ).one()
    assert document.observer == USER_NAME
    assert document.observed == USER_NAME
    assert document.workspace_name == APP_NAME
