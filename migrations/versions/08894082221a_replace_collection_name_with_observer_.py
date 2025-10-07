"""replace collection name with observer_observed

Revision ID: 08894082221a
Revises: 564ba40505c5
Create Date: 2025-10-03 16:48:13.270834

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from migrations.utils import column_exists, constraint_exists, fk_exists, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "08894082221a"
down_revision: str | None = "564ba40505c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace collections.name and documents.collection_name with observer and observed fields."""
    schema = settings.DB.SCHEMA
    inspector = sa.inspect(op.get_bind())
    connection = op.get_bind()

    # SESSION_NAME MIGRATION
    # Replace NULL session_name values with empty strings and make column non-nullable
    # This only applies to documents table

    # Update documents table
    connection.execute(
        text(
            """
            UPDATE documents
            SET session_name = ''
            WHERE session_name IS NULL
        """
        )
    )
    if column_exists("documents", "session_name", inspector):
        op.alter_column("documents", "session_name", nullable=False, schema=schema)

    # COLLECTIONS TABLE
    # Step 1: Add new observer and observed columns to collections
    if not column_exists("collections", "observer", inspector):
        op.add_column(
            "collections",
            sa.Column("observer", sa.TEXT(), nullable=True),
            schema=schema,
        )

    if not column_exists("collections", "observed", inspector):
        op.add_column(
            "collections",
            sa.Column("observed", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Step 2: Populate collections observer and observed from existing name field
    # The logic is:
    # - observer = peer_name (the exact peer ID)
    # - If name is "global_representation", observed = peer_name
    # - Otherwise, observed = name with the "observer_" prefix stripped
    connection.execute(
        text(
            """
            UPDATE collections
            SET
                observer = peer_name,
                observed = CASE
                    WHEN name = 'global_representation' THEN peer_name
                    ELSE substring(name from length(peer_name) + 2)
                END
            WHERE observer IS NULL OR observed IS NULL
        """
        )
    )

    # Step 3: Make collections observer and observed NOT NULL
    op.alter_column("collections", "observer", nullable=False, schema=schema)
    op.alter_column("collections", "observed", nullable=False, schema=schema)

    # DOCUMENTS TABLE
    # Step 4: Add new observer and observed columns to documents
    if not column_exists("documents", "observer", inspector):
        op.add_column(
            "documents",
            sa.Column("observer", sa.TEXT(), nullable=True),
            schema=schema,
        )

    if not column_exists("documents", "observed", inspector):
        op.add_column(
            "documents",
            sa.Column("observed", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Step 5: Populate documents observer and observed from collections table
    # Join to the already-populated collections table to get authoritative values
    # Process in batches of 1000 to reduce query size
    batch_size = 1000
    while True:
        result = connection.execute(
            text(
                """
                WITH batch AS (
                    SELECT d.ctid
                    FROM documents d
                    WHERE d.observer IS NULL OR d.observed IS NULL
                    LIMIT :batch_size
                )
                UPDATE documents d
                SET
                    observer = c.observer,
                    observed = c.observed
                FROM collections c, batch
                WHERE d.ctid = batch.ctid
                    AND d.collection_name = c.name
                    AND d.peer_name = c.peer_name
                    AND d.workspace_name = c.workspace_name
            """
            ),
            {"batch_size": batch_size},
        )
        if result.rowcount == 0:
            break

    # Step 6: Make documents observer and observed NOT NULL
    op.alter_column("documents", "observer", nullable=False, schema=schema)
    op.alter_column("documents", "observed", nullable=False, schema=schema)

    # CONSTRAINTS AND INDEXES
    # Step 7: Drop ALL foreign key constraints from documents that might reference collections.peer_name
    # Get all foreign keys on documents table
    documents_fks = inspector.get_foreign_keys("documents", schema=schema)
    for fk in documents_fks:
        fk_name = fk.get("name")
        # Drop any FK that references collections or peers and includes peer_name or collection_name
        if fk_name and any(
            pattern in fk_name
            for pattern in [
                "collection_name",
                "peer_name",
            ]
        ):
            op.drop_constraint(
                fk_name,
                "documents",
                type_="foreignkey",
                schema=schema,
            )

    # Step 7a: Drop ALL foreign key constraints from collections that reference peer_name
    collections_fks = inspector.get_foreign_keys("collections", schema=schema)
    for fk in collections_fks:
        fk_name = fk.get("name")
        if fk_name and "peer_name" in fk_name:
            op.drop_constraint(
                fk_name,
                "collections",
                type_="foreignkey",
                schema=schema,
            )

    # Step 7c: Drop the peer_name column from collections (observer replaces it)
    if column_exists("collections", "peer_name", inspector):
        op.drop_column("collections", "peer_name", schema=schema)

    # Step 7d: Drop the peer_name column from documents (observed replaces it)
    if column_exists("documents", "peer_name", inspector):
        op.drop_column("documents", "peer_name", schema=schema)

    # Step 8: Drop the old unique constraint on collections that includes name
    if constraint_exists(
        "collections", "unique_name_collection_peer", "unique", inspector
    ):
        op.drop_constraint(
            "unique_name_collection_peer", "collections", type_="unique", schema=schema
        )

    # Step 9: Create new unique constraint on collections (without peer_name)
    if not constraint_exists(
        "collections", "unique_observer_observed_collection", "unique", inspector
    ):
        op.create_unique_constraint(
            "unique_observer_observed_collection",
            "collections",
            ["observer", "observed", "workspace_name"],
            schema=schema,
        )

    # Step 10: Add composite foreign key constraint for observer peer on collections
    if not fk_exists(
        "collections", "collections_observer_workspace_name_fkey", inspector
    ):
        op.create_foreign_key(
            "collections_observer_workspace_name_fkey",
            "collections",
            "peers",
            ["observer", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 11: Add composite foreign key constraint for observed peer on collections
    if not fk_exists(
        "collections", "collections_observed_workspace_name_fkey", inspector
    ):
        op.create_foreign_key(
            "collections_observed_workspace_name_fkey",
            "collections",
            "peers",
            ["observed", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 12: Add composite foreign key constraint from documents to collections using observer/observed
    if not fk_exists(
        "documents", "documents_observer_observed_workspace_name_fkey", inspector
    ):
        op.create_foreign_key(
            "documents_observer_observed_workspace_name_fkey",
            "documents",
            "collections",
            ["observer", "observed", "workspace_name"],
            ["observer", "observed", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 13: Add composite foreign key constraint for observer peer on documents
    if not fk_exists("documents", "documents_observer_workspace_name_fkey", inspector):
        op.create_foreign_key(
            "documents_observer_workspace_name_fkey",
            "documents",
            "peers",
            ["observer", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 14: Add composite foreign key constraint for observed peer on documents
    if not fk_exists("documents", "documents_observed_workspace_name_fkey", inspector):
        op.create_foreign_key(
            "documents_observed_workspace_name_fkey",
            "documents",
            "peers",
            ["observed", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 15: Create indexes for observer and observed on collections
    if not index_exists("collections", "idx_collections_observer", inspector):
        op.create_index(
            "idx_collections_observer",
            "collections",
            ["observer"],
            schema=schema,
        )

    if not index_exists("collections", "idx_collections_observed", inspector):
        op.create_index(
            "idx_collections_observed",
            "collections",
            ["observed"],
            schema=schema,
        )

    # Step 16: Create indexes for observer and observed on documents
    if not index_exists("documents", "idx_documents_observer", inspector):
        op.create_index(
            "idx_documents_observer",
            "documents",
            ["observer"],
            schema=schema,
        )

    if not index_exists("documents", "idx_documents_observed", inspector):
        op.create_index(
            "idx_documents_observed",
            "documents",
            ["observed"],
            schema=schema,
        )

    # Step 17: Drop the name column from collections
    if column_exists("collections", "name", inspector):
        op.drop_column("collections", "name", schema=schema)

    # Step 18: Drop the collection_name column from documents
    if column_exists("documents", "collection_name", inspector):
        op.drop_column("documents", "collection_name", schema=schema)


def downgrade() -> None:
    """Restore collections.name and documents.collection_name from observer and observed fields."""
    schema = settings.DB.SCHEMA
    inspector = sa.inspect(op.get_bind())
    connection = op.get_bind()

    # SESSION_NAME MIGRATION ROLLBACK
    # Make session_name nullable again for documents table

    # Revert documents table
    if column_exists("documents", "session_name", inspector):
        op.alter_column("documents", "session_name", nullable=True, schema=schema)

    # COLLECTIONS TABLE
    # Step 1: Add back the name column to collections
    if not column_exists("collections", "name", inspector):
        op.add_column(
            "collections",
            sa.Column("name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Step 2: Populate collections name from observer and observed
    connection.execute(
        text(
            """
            UPDATE collections
            SET name = CASE
                WHEN observer = observed THEN 'global_representation'
                ELSE observer || '_' || observed
            END
            WHERE name IS NULL
        """
        )
    )

    # Step 3: Make collections name NOT NULL
    op.alter_column("collections", "name", nullable=False, schema=schema)

    # Step 3a: Restore peer_name column to collections (set to observer value)
    if not column_exists("collections", "peer_name", inspector):
        op.add_column(
            "collections",
            sa.Column("peer_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Populate peer_name with observer value
    connection.execute(
        text(
            """
            UPDATE collections
            SET peer_name = observer
            WHERE peer_name IS NULL
        """
        )
    )

    # Make peer_name NOT NULL
    op.alter_column("collections", "peer_name", nullable=False, schema=schema)

    # Recreate the foreign key constraint for peer_name
    if not fk_exists(
        "collections", "collections_peer_name_workspace_name_fkey", inspector
    ):
        op.create_foreign_key(
            "collections_peer_name_workspace_name_fkey",
            "collections",
            "peers",
            ["peer_name", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # DOCUMENTS TABLE
    # Step 4: Add back the collection_name column to documents
    if not column_exists("documents", "collection_name", inspector):
        op.add_column(
            "documents",
            sa.Column("collection_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Step 5: Populate documents collection_name from observer and observed
    connection.execute(
        text(
            """
            UPDATE documents
            SET collection_name = CASE
                WHEN observer = observed THEN 'global_representation'
                ELSE observer || '_' || observed
            END
            WHERE collection_name IS NULL
        """
        )
    )

    # Step 6: Make documents collection_name NOT NULL
    op.alter_column("documents", "collection_name", nullable=False, schema=schema)

    # Step 6a: Restore peer_name column to documents (set to observed value)
    if not column_exists("documents", "peer_name", inspector):
        op.add_column(
            "documents",
            sa.Column("peer_name", sa.TEXT(), nullable=True),
            schema=schema,
        )

    # Populate peer_name with observed value
    connection.execute(
        text(
            """
            UPDATE documents
            SET peer_name = observed
            WHERE peer_name IS NULL
        """
        )
    )

    # Make peer_name NOT NULL
    op.alter_column("documents", "peer_name", nullable=False, schema=schema)

    # Recreate the foreign key constraint for peer_name on documents
    if not fk_exists("documents", "documents_peer_name_workspace_name_fkey", inspector):
        op.create_foreign_key(
            "documents_peer_name_workspace_name_fkey",
            "documents",
            "peers",
            ["peer_name", "workspace_name"],
            ["name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 7: Add check constraint for name length on collections
    if not constraint_exists("collections", "name_length", "check", inspector):
        op.create_check_constraint(
            "name_length",
            "collections",
            "length(name) <= 1025",
            schema=schema,
        )

    # CONSTRAINTS AND INDEXES
    # Step 8: Drop new foreign key constraints from documents
    if fk_exists(
        "documents", "documents_observer_observed_workspace_name_fkey", inspector
    ):
        op.drop_constraint(
            "documents_observer_observed_workspace_name_fkey",
            "documents",
            type_="foreignkey",
            schema=schema,
        )

    if fk_exists("documents", "documents_observer_workspace_name_fkey", inspector):
        op.drop_constraint(
            "documents_observer_workspace_name_fkey",
            "documents",
            type_="foreignkey",
            schema=schema,
        )

    if fk_exists("documents", "documents_observed_workspace_name_fkey", inspector):
        op.drop_constraint(
            "documents_observed_workspace_name_fkey",
            "documents",
            type_="foreignkey",
            schema=schema,
        )

    # Step 9: Drop the new unique constraint on collections
    if constraint_exists(
        "collections", "unique_observer_observed_collection", "unique", inspector
    ):
        op.drop_constraint(
            "unique_observer_observed_collection",
            "collections",
            type_="unique",
            schema=schema,
        )

    # Step 10: Recreate the old unique constraint on collections
    if not constraint_exists(
        "collections", "unique_name_collection_peer", "unique", inspector
    ):
        op.create_unique_constraint(
            "unique_name_collection_peer",
            "collections",
            ["name", "peer_name", "workspace_name"],
            schema=schema,
        )

    # Step 11: Recreate the old foreign key constraint from documents to collections
    if not fk_exists(
        "documents",
        "documents_collection_name_peer_name_workspace_name_fkey",
        inspector,
    ):
        op.create_foreign_key(
            "documents_collection_name_peer_name_workspace_name_fkey",
            "documents",
            "collections",
            ["collection_name", "peer_name", "workspace_name"],
            ["name", "peer_name", "workspace_name"],
            source_schema=schema,
            referent_schema=schema,
        )

    # Step 12: Drop foreign key constraints from collections
    if fk_exists("collections", "collections_observer_workspace_name_fkey", inspector):
        op.drop_constraint(
            "collections_observer_workspace_name_fkey",
            "collections",
            type_="foreignkey",
            schema=schema,
        )

    if fk_exists("collections", "collections_observed_workspace_name_fkey", inspector):
        op.drop_constraint(
            "collections_observed_workspace_name_fkey",
            "collections",
            type_="foreignkey",
            schema=schema,
        )

    # Step 13: Drop indexes from collections
    if index_exists("collections", "idx_collections_observer", inspector):
        op.drop_index(
            "idx_collections_observer", table_name="collections", schema=schema
        )

    if index_exists("collections", "idx_collections_observed", inspector):
        op.drop_index(
            "idx_collections_observed", table_name="collections", schema=schema
        )

    # Step 14: Drop indexes from documents
    if index_exists("documents", "idx_documents_observer", inspector):
        op.drop_index("idx_documents_observer", table_name="documents", schema=schema)

    if index_exists("documents", "idx_documents_observed", inspector):
        op.drop_index("idx_documents_observed", table_name="documents", schema=schema)

    # Step 15: Drop observer and observed columns from collections
    if column_exists("collections", "observer", inspector):
        op.drop_column("collections", "observer", schema=schema)

    if column_exists("collections", "observed", inspector):
        op.drop_column("collections", "observed", schema=schema)

    # Step 16: Drop observer and observed columns from documents
    if column_exists("documents", "observer", inspector):
        op.drop_column("documents", "observer", schema=schema)

    if column_exists("documents", "observed", inspector):
        op.drop_column("documents", "observed", schema=schema)
