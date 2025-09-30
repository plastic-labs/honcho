"""migrate documents to per collection tables

Revision ID: dbafd24f50bb
Revises: 88b0fb10906f
Create Date: 2025-09-29 18:01:08.577358

"""

from collections.abc import Sequence
from logging import getLogger

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from migrations.utils import table_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "dbafd24f50bb"
down_revision: str | None = "88b0fb10906f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = getLogger(__name__)


def get_documents_table_name(collection_id: str) -> str:
    """Get the table name for a collection's documents."""
    safe_id = collection_id.replace("-", "_")
    return f"documents_{safe_id}"


def upgrade() -> None:
    """Migrate documents from single table to per-collection tables."""
    schema = settings.DB.SCHEMA
    connection = op.get_bind()

    # Get all collections
    collections_result = connection.execute(
        text(f"SELECT id, name, peer_name, workspace_name FROM {schema}.collections")
    )
    collections = collections_result.fetchall()

    logger.info(f"Found {len(collections)} collections to process")

    # For each collection, create a new table and migrate documents
    for collection_id, name, peer_name, workspace_name in collections:
        table_name = get_documents_table_name(collection_id)

        # Count documents for this collection
        count_result = connection.execute(
            text(f"""
                SELECT COUNT(*) FROM {schema}.documents
                WHERE workspace_name = :workspace_name
                AND peer_name = :peer_name
                AND collection_name = :collection_name
            """),
            {
                "workspace_name": workspace_name,
                "peer_name": peer_name,
                "collection_name": name,
            },
        )
        doc_count = count_result.scalar() or 0

        if doc_count == 0:
            logger.info(f"Skipping collection {name} - no documents")
            continue

        logger.info(f"Processing collection {name} with {doc_count} documents")

        # Create the new table
        connection.execute(
            text(f"""
                CREATE TABLE {schema}.{table_name} (
                    id TEXT PRIMARY KEY,
                    internal_metadata JSONB DEFAULT '{{}}',
                    content TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    times_derived INTEGER NOT NULL DEFAULT 1,
                    peer_name TEXT NOT NULL,
                    workspace_name TEXT NOT NULL,

                    CONSTRAINT {table_name}_id_length CHECK (length(id) = 21),
                    CONSTRAINT {table_name}_content_length CHECK (length(content) <= 65535),
                    CONSTRAINT {table_name}_id_format CHECK (id ~ '^[A-Za-z0-9_-]+$'),
                    CONSTRAINT {table_name}_fk_peer FOREIGN KEY (peer_name, workspace_name)
                        REFERENCES {schema}.peers(name, workspace_name),
                    CONSTRAINT {table_name}_fk_workspace FOREIGN KEY (workspace_name)
                        REFERENCES {schema}.workspaces(name)
                )
            """)
        )

        # Create indexes
        connection.execute(
            text(f"""
                CREATE INDEX {table_name}_created_at_idx ON {schema}.{table_name}(created_at);
                CREATE INDEX {table_name}_times_derived_idx ON {schema}.{table_name}(times_derived);
                CREATE INDEX {table_name}_peer_name_idx ON {schema}.{table_name}(peer_name);
                CREATE INDEX {table_name}_workspace_name_idx ON {schema}.{table_name}(workspace_name);
                CREATE INDEX {table_name}_embedding_hnsw ON {schema}.{table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
            """)
        )

        # Migrate documents, extracting times_derived from JSONB
        connection.execute(
            text(f"""
                INSERT INTO {schema}.{table_name}
                (id, internal_metadata, content, embedding, created_at, times_derived, peer_name, workspace_name)
                SELECT
                    id,
                    internal_metadata,
                    content,
                    embedding,
                    created_at,
                    COALESCE((internal_metadata->>'times_derived')::integer, 1) as times_derived,
                    peer_name,
                    workspace_name
                FROM {schema}.documents
                WHERE workspace_name = :workspace_name
                AND peer_name = :peer_name
                AND collection_name = :collection_name
            """),
            {
                "workspace_name": workspace_name,
                "peer_name": peer_name,
                "collection_name": name,
            },
        )

        logger.info(f"Migrated {doc_count} documents to {table_name}")

    # Drop the old documents table
    inspector = sa.inspect(connection)
    if table_exists("documents", inspector):
        logger.warning("Dropping old documents table")
        op.drop_table("documents", schema=schema)


def downgrade() -> None:
    """Restore documents from per-collection tables back to single table."""
    schema = settings.DB.SCHEMA
    connection = op.get_bind()

    # Recreate the old documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.TEXT(), nullable=False),
        sa.Column("internal_metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("content", sa.TEXT(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("collection_name", sa.TEXT(), nullable=False),
        sa.Column("peer_name", sa.TEXT(), nullable=False),
        sa.Column("workspace_name", sa.TEXT(), nullable=False),
        sa.CheckConstraint("length(id) = 21", name="documents_id_length"),
        sa.CheckConstraint("length(content) <= 65535", name="documents_content_length"),
        sa.CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="documents_id_format"),
        sa.ForeignKeyConstraint(
            ["collection_name", "peer_name", "workspace_name"],
            [
                f"{schema}.collections.name",
                f"{schema}.collections.peer_name",
                f"{schema}.collections.workspace_name",
            ],
            name="documents_fk_collection",
        ),
        sa.ForeignKeyConstraint(
            ["peer_name", "workspace_name"],
            [f"{schema}.peers.name", f"{schema}.peers.workspace_name"],
            name="documents_fk_peer",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_name"],
            [f"{schema}.workspaces.name"],
            name="documents_fk_workspace",
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=schema,
    )

    # Create indexes on the old table
    op.create_index(
        "documents_created_at_idx", "documents", ["created_at"], schema=schema
    )
    op.create_index(
        "documents_collection_name_idx", "documents", ["collection_name"], schema=schema
    )
    op.create_index(
        "documents_peer_name_idx", "documents", ["peer_name"], schema=schema
    )
    op.create_index(
        "documents_workspace_name_idx", "documents", ["workspace_name"], schema=schema
    )

    # Create HNSW index on embedding column
    connection.execute(
        text(f"""
            CREATE INDEX documents_embedding_hnsw ON {schema}.documents
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        """)
    )

    # Get all collections
    collections_result = connection.execute(
        text(f"SELECT id, name, peer_name, workspace_name FROM {schema}.collections")
    )
    collections = collections_result.fetchall()

    # Migrate documents back from per-collection tables
    for collection_id, name, _peer_name, _workspace_name in collections:
        table_name = get_documents_table_name(collection_id)

        # Check if the table exists
        inspector = sa.inspect(connection)
        if not inspector.has_table(table_name, schema=schema):
            continue

        # Migrate documents back (adding collection_name back and times_derived to JSONB)
        connection.execute(
            text(f"""
                INSERT INTO {schema}.documents
                (id, internal_metadata, content, embedding, created_at, collection_name, peer_name, workspace_name)
                SELECT
                    id,
                    jsonb_set(internal_metadata, '{{times_derived}}', to_jsonb(times_derived)),
                    content,
                    embedding,
                    created_at,
                    :collection_name,
                    peer_name,
                    workspace_name
                FROM {schema}.{table_name}
            """),
            {"collection_name": name},
        )

        # Drop the per-collection table
        connection.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name} CASCADE"))

        logger.info(f"Restored documents from {table_name}")
