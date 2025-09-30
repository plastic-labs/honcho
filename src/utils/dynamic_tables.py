"""Utilities for managing dynamic per-collection document tables."""

import datetime
from typing import Any, TypeVar

from nanoid import generate as generate_nanoid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm.properties import MappedColumn
from sqlalchemy.sql import func, text

from src.db import Base

# Registry for dynamic models
dynamic_model_registry: dict[str, type] = {}

T = TypeVar("T")


def get_documents_table_name(collection_id: str) -> str:
    """
    Get the table name for a collection's documents.

    Note: Collection IDs are generated without hyphens to ensure valid PostgreSQL identifiers.
    For backward compatibility with existing collections that may have hyphens in their IDs,
    hyphens are replaced with underscores.

    Returns lowercase table name to match PostgreSQL's default behavior when creating
    tables without quotes.
    """
    # Replace hyphens with underscores for backward compatibility
    safe_id = collection_id.replace("-", "_").lower()
    return f"documents_{safe_id}"


def create_dynamic_document_model(collection_id: str) -> type:
    """
    Create a dynamic Document model class for a specific collection.

    Args:
        collection_id: The ID of the collection

    Returns:
        A SQLAlchemy model class for the collection's documents table
    """
    # Check cache first
    if collection_id in dynamic_model_registry:
        return dynamic_model_registry[collection_id]

    table_name = get_documents_table_name(collection_id)

    # Create the class dynamically
    class DynamicDocument(Base):
        __tablename__ = table_name
        __table_args__ = (
            CheckConstraint("length(id) = 21", name=f"{table_name}_id_length"),
            CheckConstraint(
                "length(content) <= 65535", name=f"{table_name}_content_length"
            ),
            CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name=f"{table_name}_id_format"),
            # Composite foreign key constraint for peers
            ForeignKeyConstraint(
                ["peer_name", "workspace_name"],
                ["peers.name", "peers.workspace_name"],
                name=f"{table_name}_fk_peer",
            ),
            # HNSW index on embedding column
            Index(
                f"{table_name}_embedding_hnsw",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )

        id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
        internal_metadata: Mapped[dict[str, Any]] = mapped_column(
            "internal_metadata", JSONB, default=dict
        )
        content: Mapped[str] = mapped_column(TEXT)
        embedding: MappedColumn[Any] = mapped_column(Vector(1536))
        created_at: Mapped[datetime.datetime] = mapped_column(
            DateTime(timezone=True), index=True, default=func.now()
        )
        times_derived: Mapped[int] = mapped_column(
            Integer, index=True, default=1, nullable=False
        )

        peer_name: Mapped[str] = mapped_column(TEXT, index=True)
        workspace_name: Mapped[str] = mapped_column(
            ForeignKey("workspaces.name"), index=True
        )

    # Cache the model
    dynamic_model_registry[collection_id] = DynamicDocument

    return DynamicDocument


async def table_exists(db: AsyncSession, collection_id: str) -> bool:
    """
    Check if a documents table exists for a collection.

    Args:
        db: Database session
        collection_id: The ID of the collection

    Returns:
        True if the table exists, False otherwise
    """
    table_name = get_documents_table_name(collection_id)
    # Query the table directly to see if it exists - this will see uncommitted changes
    # in the current transaction as well as committed tables
    check_sql = text(f"SELECT to_regclass('public.{table_name}') IS NOT NULL;")
    result = await db.execute(check_sql)
    exists = result.scalar() or False
    return exists


async def create_documents_table(db: AsyncSession, collection_id: str) -> None:
    """
    Create a new documents table for a collection.

    Args:
        db: Database session
        collection_id: The ID of the collection
    """
    table_name = get_documents_table_name(collection_id)

    # Create the table first
    create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY DEFAULT '',
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
                REFERENCES peers(name, workspace_name),
            CONSTRAINT {table_name}_fk_workspace FOREIGN KEY (workspace_name)
                REFERENCES workspaces(name)
        );
    """)

    await db.execute(create_table_sql)
    await db.flush()

    # Create indexes separately to ensure they're applied after table creation
    create_indexes_sql = text(f"""
        CREATE INDEX IF NOT EXISTS {table_name}_created_at_idx ON {table_name}(created_at);
        CREATE INDEX IF NOT EXISTS {table_name}_times_derived_idx ON {table_name}(times_derived);
        CREATE INDEX IF NOT EXISTS {table_name}_peer_name_idx ON {table_name}(peer_name);
        CREATE INDEX IF NOT EXISTS {table_name}_workspace_name_idx ON {table_name}(workspace_name);
        CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw ON {table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    """)

    await db.execute(create_indexes_sql)
    await db.flush()
    # Note: Caller is responsible for committing the transaction


async def drop_documents_table(db: AsyncSession, collection_id: str) -> None:
    """
    Drop a documents table for a collection.

    Args:
        db: Database session
        collection_id: The ID of the collection
    """
    table_name = get_documents_table_name(collection_id)

    # Drop the table
    drop_table_sql = text(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    await db.execute(drop_table_sql)
    # Note: Caller is responsible for committing the transaction

    # Remove from cache
    if collection_id in dynamic_model_registry:
        del dynamic_model_registry[collection_id]
