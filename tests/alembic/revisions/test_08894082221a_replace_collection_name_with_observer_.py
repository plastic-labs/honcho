"""Hooks for revision 08894082221a (observer/observed refactor)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import inspect, text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

COLLECTION_ID: str = generate_nanoid()
GLOBAL_REP_COLLECTION_NAME = "global_representation"
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

    # Create collection with the old schema (has 'name' field)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."collections" '
            + '("id", "name", "peer_name", "workspace_name") '
            + "VALUES (:collection_id, :collection_name, :peer_name, :workspace_name)"
        ),
        {
            "collection_id": COLLECTION_ID,
            "collection_name": GLOBAL_REP_COLLECTION_NAME,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    # Create document with the old schema (has 'collection_name' field) without session_name
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "collection_name", "peer_name", "workspace_name", "content") '
            + "VALUES (:document_id, :collection_name, :peer_name, :workspace_name, :content)"
        ),
        {
            "document_id": DOCUMENT_ID,
            "collection_name": GLOBAL_REP_COLLECTION_NAME,
            "peer_name": LEGACY_PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
            "content": "test content",
        },
    )

    # Bulk create many documents with the old schema (no session_name)
    bulk_stmt = text(
        f'INSERT INTO "{schema}"."documents" '
        + '("id", "collection_name", "peer_name", "workspace_name", "content") '
        + "VALUES (:document_id, :collection_name, :peer_name, :workspace_name, :content)"
    )

    params_template = {
        "collection_name": GLOBAL_REP_COLLECTION_NAME,
        "peer_name": LEGACY_PEER_NAME,
        "workspace_name": WORKSPACE_NAME,
        "content": "test content",
    }

    total_docs = 120_000
    batch_size = 10_000
    for start in range(0, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        batch_params = [
            {"document_id": generate_nanoid(), **params_template}
            for _ in range(start, end)
        ]

        conn.execute(bulk_stmt, batch_params)

    # Speed up bulk seeding
    conn.execute(text("SET LOCAL synchronous_commit = OFF"))

    # -----------------------------
    # Group 1: self-observation collections (global_representation)
    # observer = peer_name, observed = peer_name
    # -----------------------------
    peer_insert_stmt = text(
        f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
        + "VALUES (:peer_id, :peer_name, :workspace_name)"
    )
    collection_insert_stmt = text(
        f'INSERT INTO "{schema}"."collections" ("id", "name", "peer_name", "workspace_name") '
        + "VALUES (:collection_id, :name, :peer_name, :workspace_name)"
    )

    total = 100_000
    batch_size = 10_000

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Insert peers for group 1
        peers_params = [
            {
                "peer_id": generate_nanoid(),
                "peer_name": f"selfpeer_{i}",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(peer_insert_stmt, peers_params)

        # Insert collections for group 1
        collections_params = [
            {
                "collection_id": generate_nanoid(),
                "name": GLOBAL_REP_COLLECTION_NAME,
                "peer_name": f"selfpeer_{i}",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(collection_insert_stmt, collections_params)

    # -----------------------------
    # Group 2: prefix pattern (observer_observed)
    # observer = 'prefix_observer', observed extracted from name suffix
    # -----------------------------
    # Create the fixed observer peer
    conn.execute(
        peer_insert_stmt,
        {
            "peer_id": generate_nanoid(),
            "peer_name": "prefix_observer",
            "workspace_name": WORKSPACE_NAME,
        },
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Observed peers for group 2
        peers_params = [
            {
                "peer_id": generate_nanoid(),
                "peer_name": f"obs_prefix_{i}",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(peer_insert_stmt, peers_params)

        # Collections for group 2 (name = 'prefix_observer_obs_prefix_<i>')
        collections_params = [
            {
                "collection_id": generate_nanoid(),
                "name": f"prefix_observer_obs_prefix_{i}",
                "peer_name": "prefix_observer",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(collection_insert_stmt, collections_params)

    # -----------------------------
    # Group 3: suffix pattern (observed_observer)
    # observer = 'suffix_observer', observed extracted from name prefix
    # -----------------------------
    # Create the fixed observer peer
    conn.execute(
        peer_insert_stmt,
        {
            "peer_id": generate_nanoid(),
            "peer_name": "suffix_observer",
            "workspace_name": WORKSPACE_NAME,
        },
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Observed peers for group 3
        peers_params = [
            {
                "peer_id": generate_nanoid(),
                "peer_name": f"obs_suffix_{i}",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(peer_insert_stmt, peers_params)

        # Collections for group 3 (name = 'obs_suffix_<i>_suffix_observer')
        collections_params = [
            {
                "collection_id": generate_nanoid(),
                "name": f"obs_suffix_{i}_suffix_observer",
                "peer_name": "suffix_observer",
                "workspace_name": WORKSPACE_NAME,
            }
            for i in range(start, end)
        ]
        conn.execute(collection_insert_stmt, collections_params)


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
            'SELECT "observer", "observed", "workspace_name", "session_name" '
            + f'FROM "{verifier.schema}"."documents" '
            + 'WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_ID},
    ).one()
    assert document.observer == LEGACY_PEER_NAME
    assert document.observed == EXPECTED_OBSERVED
    assert document.workspace_name == WORKSPACE_NAME
    assert document.session_name == "__global_observations__"

    # Verify no NULLs remain in documents.session_name
    verifier.assert_no_nulls("documents", "session_name")

    # Verify collections mapping for Group 1 (self-observation)
    count_self = verifier.conn.execute(
        text(
            "SELECT COUNT(1) FROM "
            + f'"{verifier.schema}"."collections" '
            + 'WHERE "workspace_name" = :ws '
            + 'AND "observer" LIKE :prefix '
            + 'AND "observed" = "observer"'
        ),
        {"ws": WORKSPACE_NAME, "prefix": "selfpeer_%"},
    ).scalar()
    assert count_self == 100_000

    # Verify collections mapping for Group 2 (prefix pattern observer_observed)
    count_prefix = verifier.conn.execute(
        text(
            "SELECT COUNT(1) FROM "
            + f'"{verifier.schema}"."collections" '
            + 'WHERE "workspace_name" = :ws '
            + 'AND "observer" = :observer '
            + 'AND "observed" LIKE :obs_prefix'
        ),
        {
            "ws": WORKSPACE_NAME,
            "observer": "prefix_observer",
            "obs_prefix": "obs_prefix_%",
        },
    ).scalar()
    assert count_prefix == 100_000

    distinct_prefix_observed = verifier.conn.execute(
        text(
            'SELECT COUNT(DISTINCT "observed") FROM '
            + f'"{verifier.schema}"."collections" '
            + 'WHERE "workspace_name" = :ws AND "observer" = :observer'
        ),
        {"ws": WORKSPACE_NAME, "observer": "prefix_observer"},
    ).scalar()
    assert distinct_prefix_observed == 100_000

    # Verify collections mapping for Group 3 (suffix pattern observed_observer)
    count_suffix = verifier.conn.execute(
        text(
            "SELECT COUNT(1) FROM "
            + f'"{verifier.schema}"."collections" '
            + 'WHERE "workspace_name" = :ws '
            + 'AND "observer" = :observer '
            + 'AND "observed" LIKE :obs_prefix'
        ),
        {
            "ws": WORKSPACE_NAME,
            "observer": "suffix_observer",
            "obs_prefix": "obs_suffix_%",
        },
    ).scalar()
    assert count_suffix == 100_000

    distinct_suffix_observed = verifier.conn.execute(
        text(
            'SELECT COUNT(DISTINCT "observed") FROM '
            + f'"{verifier.schema}"."collections" '
            + 'WHERE "workspace_name" = :ws AND "observer" = :observer'
        ),
        {"ws": WORKSPACE_NAME, "observer": "suffix_observer"},
    ).scalar()
    assert distinct_suffix_observed == 100_000
