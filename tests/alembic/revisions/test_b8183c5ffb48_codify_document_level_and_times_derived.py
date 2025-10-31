"""Hooks for revision b8183c5ffb48 (codify_document_level_and_times_derived)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Test data constants
WORKSPACE_NAME = "test-workspace"
OBSERVER_NAME = "test-observer"
OBSERVED_NAME = "test-observed"
SESSION_NAME = "test-session"


@register_before_upgrade("b8183c5ffb48")
def prepare_codify_document_level_and_times_derived(
    verifier: MigrationVerifier,
) -> None:
    """Seed state and assertions before upgrading to b8183c5ffb48."""
    # Verify columns don't exist yet
    verifier.assert_column_exists("documents", "level", exists=False)
    verifier.assert_column_exists("documents", "times_derived", exists=False)

    # Verify CHECK constraint doesn't exist yet
    verifier.assert_constraint_exists("documents", "level_valid", "check", exists=False)

    conn = verifier.conn
    schema = verifier.schema

    # Create workspace
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" ("id", "name") VALUES (:ws_id, :ws_name)'
        ),
        {"ws_id": generate_nanoid(), "ws_name": WORKSPACE_NAME},
    )

    # Create peers (observer and observed)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name") '
            + "VALUES (:observer_id, :observer_name, :ws_name), "
            + "(:observed_id, :observed_name, :ws_name)"
        ),
        {
            "observer_id": generate_nanoid(),
            "observer_name": OBSERVER_NAME,
            "ws_name": WORKSPACE_NAME,
            "observed_id": generate_nanoid(),
            "observed_name": OBSERVED_NAME,
        },
    )

    # Create session
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name") '
            + "VALUES (:session_id, :session_name, :ws_name)"
        ),
        {
            "session_id": generate_nanoid(),
            "session_name": SESSION_NAME,
            "ws_name": WORKSPACE_NAME,
        },
    )

    # Create collection (required for documents)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."collections" ("id", "observer", "observed", "workspace_name") '
            + "VALUES (:collection_id, :observer, :observed, :ws_name)"
        ),
        {
            "collection_id": generate_nanoid(),
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
        },
    )

    # Create documents with different scenarios
    # Scenario 1: Document with both level and times_derived in internal_metadata
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": generate_nanoid(),
            "content": "Document with explicit level and times_derived=3",
            "metadata": json.dumps({"level": "explicit", "times_derived": 3}),
            "embedding": [0.1] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Scenario 2: Document with deductive level in internal_metadata
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": generate_nanoid(),
            "content": "Document with deductive level and times_derived=5",
            "metadata": json.dumps({"level": "deductive", "times_derived": 5}),
            "embedding": [0.2] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Scenario 3: Document without level or times_derived (should get defaults)
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": generate_nanoid(),
            "content": "Document without level or times_derived fields",
            "metadata": json.dumps({"other_field": "value"}),
            "embedding": [0.3] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Scenario 4: Document with only level in metadata
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": generate_nanoid(),
            "content": "Document with only level field",
            "metadata": json.dumps({"level": "explicit"}),
            "embedding": [0.4] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Scenario 5: Document with only times_derived in metadata
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": generate_nanoid(),
            "content": "Document with only times_derived field",
            "metadata": json.dumps({"times_derived": 7}),
            "embedding": [0.5] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Verify we have exactly 5 documents
    count = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."documents"')).scalar()
    assert count == 5, f"Expected 5 documents but found {count}"


@register_after_upgrade("b8183c5ffb48")
def verify_codify_document_level_and_times_derived(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of b8183c5ffb48."""
    # Verify columns were added with correct nullability
    verifier.assert_column_exists("documents", "level", nullable=False)
    verifier.assert_column_exists("documents", "times_derived", nullable=False)

    # Verify CHECK constraint exists
    verifier.assert_constraint_exists("documents", "level_valid", "check")

    conn = verifier.conn
    schema = verifier.schema

    # Verify all rows have non-null values after migration
    verifier.assert_no_nulls("documents", "level")
    verifier.assert_no_nulls("documents", "times_derived")

    # Verify data transformation: level extracted from internal_metadata
    explicit_count = conn.execute(
        text(f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE "level" = :level'),
        {"level": "explicit"},
    ).scalar()
    deductive_count = conn.execute(
        text(f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE "level" = :level'),
        {"level": "deductive"},
    ).scalar()

    # 3 documents should have explicit (scenarios 1, 3, 4)
    # 1 document should have deductive (scenario 2)
    # 1 document should have default explicit (scenario 5)
    assert explicit_count == 4, f"Expected 4 explicit documents, got {explicit_count}"
    assert deductive_count == 1, f"Expected 1 deductive document, got {deductive_count}"

    # Verify specific times_derived values were migrated correctly
    times_derived_3 = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."documents" '
            + 'WHERE "times_derived" = 3 AND "content" LIKE :pattern'
        ),
        {"pattern": "%explicit level and times_derived=3%"},
    ).scalar()
    assert (
        times_derived_3 == 1
    ), f"Expected 1 document with times_derived=3, got {times_derived_3}"

    times_derived_5 = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."documents" '
            + 'WHERE "times_derived" = 5 AND "content" LIKE :pattern'
        ),
        {"pattern": "%deductive level and times_derived=5%"},
    ).scalar()
    assert (
        times_derived_5 == 1
    ), f"Expected 1 document with times_derived=5, got {times_derived_5}"

    times_derived_7 = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."documents" '
            + 'WHERE "times_derived" = 7 AND "content" LIKE :pattern'
        ),
        {"pattern": "%only times_derived field%"},
    ).scalar()
    assert (
        times_derived_7 == 1
    ), f"Expected 1 document with times_derived=7, got {times_derived_7}"

    # Verify default times_derived=1 was applied to documents without it
    times_derived_1 = conn.execute(
        text(f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE "times_derived" = 1')
    ).scalar()
    assert (
        times_derived_1 == 2
    ), f"Expected 2 documents with times_derived=1, got {times_derived_1}"

    # Verify internal_metadata still contains the original data (NOT removed by migration)
    level_in_metadata = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE internal_metadata ? \'level\''
        )
    ).scalar()
    # 3 documents had level in metadata (scenarios 1, 2, 4)
    assert (
        level_in_metadata == 3
    ), f"Expected 3 documents with level in metadata, got {level_in_metadata}"

    times_derived_in_metadata = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."documents" WHERE internal_metadata ? \'times_derived\''
        )
    ).scalar()
    # 3 documents had times_derived in metadata (scenarios 1, 2, 5)
    assert (
        times_derived_in_metadata == 3
    ), f"Expected 3 documents with times_derived in metadata, got {times_derived_in_metadata}"

    # Verify server defaults work for new documents
    new_doc_id = generate_nanoid()
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."documents" '
            + '("id", "content", "internal_metadata", "embedding", "observer", "observed", "workspace_name", "session_name") '
            + "VALUES (:id, :content, :metadata, :embedding, :observer, :observed, :ws_name, :session_name)"
        ),
        {
            "id": new_doc_id,
            "content": "New document after migration",
            "metadata": json.dumps({}),
            "embedding": [0.6] * 1536,
            "observer": OBSERVER_NAME,
            "observed": OBSERVED_NAME,
            "ws_name": WORKSPACE_NAME,
            "session_name": SESSION_NAME,
        },
    )

    # Verify the new document got default values
    new_doc = conn.execute(
        text(
            f'SELECT "level", "times_derived" FROM "{schema}"."documents" WHERE "id" = :id'
        ),
        {"id": new_doc_id},
    ).one()
    assert (
        new_doc.level == "explicit"
    ), f"Expected new document to have level='explicit', got {new_doc.level}"
    assert (
        new_doc.times_derived == 1
    ), f"Expected new document to have times_derived=1, got {new_doc.times_derived}"
