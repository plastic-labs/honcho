"""
PostgreSQL pgvector vector store implementation.

This module provides a pgvector-based implementation of the VectorStore interface
using the existing embedding columns on documents and message_embeddings tables.
"""

import logging
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dependencies import tracked_db

from . import VectorQueryResult, VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)


class PgVectorStore(VectorStore):
    """
    PostgreSQL pgvector implementation of the VectorStore interface.

    Uses the existing embedding columns on documents and message_embeddings tables,
    providing transactional consistency with document/message metadata.

    Namespace mapping:
    - {prefix}.{workspace}.{observer}.{observed} -> documents.embedding
    - {prefix}.{workspace}.messages -> message_embeddings.embedding
    """

    def __init__(self):
        """Initialize the pgvector store."""
        super().__init__()

    def _parse_namespace(self, namespace: str) -> tuple[str, dict[str, str]]:
        """
        Parse a namespace string to determine the table and filter context.

        Args:
            namespace: Namespace string like "{prefix}.{workspace}.{observer}.{observed}"
                       or "{prefix}.{workspace}.messages"

        Returns:
            Tuple of (table_type, context_dict) where:
            - table_type is "documents" or "message_embeddings"
            - context_dict contains workspace_name and optionally observer/observed
        """
        parts = namespace.split(".")

        # Expected formats:
        # {prefix}.{workspace}.messages -> message_embeddings
        # {prefix}.{workspace}.{observer}.{observed} -> documents
        if len(parts) < 3:
            raise ValueError(f"Invalid namespace format: {namespace}")

        workspace = parts[1]

        if parts[2] == "messages":
            return "message_embeddings", {"workspace_name": workspace}
        elif len(parts) >= 4:
            observer = parts[2]
            observed = parts[3]
            return "documents", {
                "workspace_name": workspace,
                "observer": observer,
                "observed": observed,
            }
        else:
            raise ValueError(f"Invalid namespace format: {namespace}")

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        """
        Upsert multiple vectors into the database.

        NOTE: This is a no-op. When pgvector is being used (as primary or secondary),
        embeddings are written directly to postgres via the ORM (in message.py/document.py).
        This method exists only to satisfy the VectorStore interface.

        The vector store abstraction is used for:
        - Queries (which still go through pgvector)
        - Writing to secondary stores (e.g., turbopuffer during migration)

        Args:
            namespace: The namespace (determines table)
            vectors: List of VectorRecord objects to upsert
        """
        # No-op: embeddings are already written to postgres via ORM
        if vectors:
            logger.debug(
                f"PgVectorStore.upsert_many() no-op for {len(vectors)} vectors in {namespace} (embeddings written via ORM)"
            )
        return VectorUpsertResult(primary_ok=True)

    async def query(
        self,
        namespace: str,
        embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[VectorQueryResult]:
        """
        Query for similar vectors using pgvector cosine distance.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)
        """
        table_type, context = self._parse_namespace(namespace)

        async with tracked_db("pgvector_query") as db:
            try:
                if table_type == "documents":
                    results = await self._query_documents(
                        db, context, embedding, top_k, filters, max_distance
                    )
                elif table_type == "message_embeddings":
                    results = await self._query_message_embeddings(
                        db, context, embedding, top_k, filters, max_distance
                    )
                else:
                    results = []

                logger.debug(
                    f"Query returned {len(results)} results from namespace {namespace}"
                )
                return results

            except Exception:
                logger.exception(f"Failed to query namespace {namespace}")
                raise

    async def _query_documents(
        self,
        db: AsyncSession,
        context: dict[str, str],
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        max_distance: float | None,
    ) -> list[VectorQueryResult]:
        """Query documents table for similar vectors."""
        # Build the query with cosine distance
        # pgvector uses <=> for cosine distance
        stmt = (
            select(
                models.Document.id,
                models.Document.embedding.cosine_distance(embedding).label("distance"),
                models.Document.workspace_name,
                models.Document.observer,
                models.Document.observed,
                models.Document.session_name,
                models.Document.level,
            )
            .where(models.Document.embedding.isnot(None))
            .where(models.Document.deleted_at.is_(None))
            .where(models.Document.workspace_name == context["workspace_name"])
            .where(models.Document.observer == context["observer"])
            .where(models.Document.observed == context["observed"])
        )

        # Apply additional filters
        if filters:
            if "session_name" in filters:
                stmt = stmt.where(
                    models.Document.session_name == filters["session_name"]
                )
            if "level" in filters:
                stmt = stmt.where(models.Document.level == filters["level"])

        # Apply max_distance filter
        if max_distance is not None:
            stmt = stmt.where(
                models.Document.embedding.cosine_distance(embedding) <= max_distance
            )

        # Order by distance and limit
        stmt = stmt.order_by("distance").limit(top_k)

        result = await db.execute(stmt)
        rows = result.all()

        return [
            VectorQueryResult(
                id=str(row.id),
                score=float(row.distance),
                metadata={
                    "workspace_name": row.workspace_name,
                    "observer": row.observer,
                    "observed": row.observed,
                    "session_name": row.session_name,
                    "level": row.level,
                },
            )
            for row in rows
        ]

    async def _query_message_embeddings(
        self,
        db: AsyncSession,
        context: dict[str, str],
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        max_distance: float | None,
    ) -> list[VectorQueryResult]:
        """Query message_embeddings table for similar vectors."""
        # Build the query with cosine distance
        stmt = (
            select(
                models.MessageEmbedding.id,
                models.MessageEmbedding.embedding.cosine_distance(embedding).label(
                    "distance"
                ),
                models.MessageEmbedding.message_id,
                models.MessageEmbedding.workspace_name,
                models.MessageEmbedding.session_name,
                models.MessageEmbedding.peer_name,
            )
            .where(models.MessageEmbedding.embedding.isnot(None))
            .where(models.MessageEmbedding.workspace_name == context["workspace_name"])
        )

        # Apply additional filters
        if filters:
            if "session_name" in filters:
                stmt = stmt.where(
                    models.MessageEmbedding.session_name == filters["session_name"]
                )
            if "peer_name" in filters:
                stmt = stmt.where(
                    models.MessageEmbedding.peer_name == filters["peer_name"]
                )
            if "message_id" in filters:
                stmt = stmt.where(
                    models.MessageEmbedding.message_id == filters["message_id"]
                )

        # Apply max_distance filter
        if max_distance is not None:
            stmt = stmt.where(
                models.MessageEmbedding.embedding.cosine_distance(embedding)
                <= max_distance
            )

        # Order by distance and limit
        stmt = stmt.order_by("distance").limit(top_k)

        result = await db.execute(stmt)
        rows = result.all()

        return [
            VectorQueryResult(
                id=str(row.id),
                score=float(row.distance),
                metadata={
                    "embedding_id": row.id,
                    "message_id": row.message_id,
                    "workspace_name": row.workspace_name,
                    "session_name": row.session_name,
                    "peer_name": row.peer_name,
                },
            )
            for row in rows
        ]

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete vectors by removing the rows from the database.

        For pgvector, since the vector data is stored in the same table as the entities,
        deleting from the vector store means deleting the actual rows.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        table_type, _ = self._parse_namespace(namespace)

        async with tracked_db("pgvector_delete") as db:
            try:
                if table_type == "documents":
                    stmt = delete(models.Document).where(models.Document.id.in_(ids))
                    await db.execute(stmt)

                elif table_type == "message_embeddings":
                    for vector_id in ids:
                        try:
                            embedding_id = int(vector_id)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid message vector id format: {vector_id}"
                            ) from exc

                        stmt = delete(models.MessageEmbedding).where(
                            models.MessageEmbedding.id == embedding_id
                        )
                        await db.execute(stmt)

                await db.commit()
                logger.debug(
                    f"Deleted {len(ids)} rows from {table_type} in namespace {namespace}"
                )

            except Exception:
                await db.rollback()
                logger.exception(
                    f"Failed to delete {len(ids)} rows from namespace {namespace}"
                )
                raise

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace by removing rows from the database.

        For pgvector, since the vector data is stored in the same table as the entities,
        deleting a namespace means deleting the actual rows.

        Args:
            namespace: The namespace to delete
        """
        table_type, context = self._parse_namespace(namespace)

        async with tracked_db("pgvector_delete_namespace") as db:
            try:
                if table_type == "documents":
                    stmt = (
                        delete(models.Document)
                        .where(
                            models.Document.workspace_name == context["workspace_name"]
                        )
                        .where(models.Document.observer == context["observer"])
                        .where(models.Document.observed == context["observed"])
                    )
                    await db.execute(stmt)

                elif table_type == "message_embeddings":
                    stmt = delete(models.MessageEmbedding).where(
                        models.MessageEmbedding.workspace_name
                        == context["workspace_name"]
                    )
                    await db.execute(stmt)

                await db.commit()
                logger.debug(f"Deleted all rows from namespace {namespace}")

            except Exception:
                await db.rollback()
                logger.exception(f"Failed to delete namespace {namespace}")
                raise

    async def close(self) -> None:
        """Close the pgvector store (no-op for pgvector)"""
        pass
