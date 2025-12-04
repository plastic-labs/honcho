"""
LanceDB vector store implementation.

This module provides a LanceDB-based implementation of the VectorStore interface
for use in self-hosted deployments of Honcho.
"""

import logging
from typing import Any

import lancedb
import pyarrow as pa

from src.config import settings

from . import QueryResult, VectorRecord, VectorStore

logger = logging.getLogger(__name__)

# Schema for LanceDB tables
# id: string, vector: fixed_size_list of float32 (1536 dimensions for OpenAI embeddings)
# Additional metadata columns are added dynamically
VECTOR_DIMENSION = 1536


class LanceDBVectorStore(VectorStore):
    """
    LanceDB implementation of the VectorStore interface.

    Uses LanceDB's embedded mode for local vector storage.
    Each namespace corresponds to a LanceDB table.
    """

    _db: lancedb.DBConnection

    def __init__(self):
        """Initialize the LanceDB vector store."""
        super().__init__()
        self._db = lancedb.connect(settings.VECTOR_STORE.LANCEDB_PATH)

    def _get_table(self, namespace: str) -> lancedb.table.Table | None:
        """Get a table if it exists, otherwise return None."""
        if namespace in self._db.table_names():
            return self._db.open_table(namespace)
        return None

    def _get_or_create_table(
        self, namespace: str, sample_data: list[dict[str, Any]] | None = None
    ) -> lancedb.table.Table:
        """
        Get existing table or create if not exists.

        Args:
            namespace: Table name (namespace)
            sample_data: Optional sample data to infer schema from

        Returns:
            LanceDB table
        """
        if namespace in self._db.table_names():
            return self._db.open_table(namespace)

        # Create table with sample data if provided
        if sample_data:
            return self._db.create_table(namespace, data=sample_data)

        # Create empty table with base schema
        schema = pa.schema(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            [
                pa.field("id", pa.string()),  # pyright: ignore[reportUnknownMemberType]
                pa.field("vector", pa.list_(pa.float32(), VECTOR_DIMENSION)),  # pyright: ignore[reportUnknownMemberType]
            ]
        )
        return self._db.create_table(namespace, schema=schema)  # pyright: ignore[reportUnknownArgumentType]

    def _row_to_dict(self, vector: VectorRecord) -> dict[str, Any]:
        """Convert a VectorRecord to a dict for LanceDB."""
        row: dict[str, Any] = {
            "id": vector.id,
            "vector": vector.embedding,
        }
        # Add metadata fields
        if vector.metadata:
            row.update(vector.metadata)
        return row

    async def upsert(
        self,
        namespace: str,
        vector: VectorRecord,
    ) -> None:
        """
        Upsert a single vector into LanceDB.

        Args:
            namespace: The namespace (table) to store the vector in
            vector: VectorRecord containing id, embedding, and metadata
        """
        try:
            row = self._row_to_dict(vector)
            table = self._get_or_create_table(namespace, sample_data=[row])

            # Use merge_insert for upsert behavior
            table.merge_insert("id").when_matched_update_all().execute([row])

            logger.debug(f"Upserted vector {vector.id} to namespace {namespace}")
        except Exception:
            logger.exception(
                f"Failed to upsert vector {vector.id} to namespace {namespace}"
            )
            raise

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> None:
        """
        Upsert multiple vectors into LanceDB.

        Args:
            namespace: The namespace (table) to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        if not vectors:
            return

        try:
            rows = [self._row_to_dict(v) for v in vectors]
            table = self._get_or_create_table(namespace, sample_data=rows)

            # Use merge_insert for upsert behavior
            table.merge_insert("id").when_matched_update_all().execute(rows)

            logger.debug(f"Upserted {len(vectors)} vectors to namespace {namespace}")
        except Exception:
            logger.exception(
                f"Failed to upsert {len(vectors)} vectors to namespace {namespace}"
            )
            raise

    async def query(
        self,
        namespace: str,
        embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[QueryResult]:
        """
        Query for similar vectors in LanceDB.

        Args:
            namespace: The namespace (table) to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of QueryResult objects, ordered by similarity (most similar first)
        """
        table = self._get_table(namespace)
        if table is None:
            logger.debug(f"Table {namespace} does not exist, returning empty results")
            return []

        try:
            # Build query (LanceDB types are incomplete, so type checker reports false positive)
            query = table.search(embedding).distance_type("cosine").limit(top_k)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]

            # Apply filters if provided
            if filters:
                where_clause = self._build_where_clause(filters)
                if where_clause:
                    query = query.where(where_clause)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            # Execute query
            results = query.to_list()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

            # Convert to QueryResult objects
            query_results: list[QueryResult] = []
            for row in results:  # pyright: ignore[reportUnknownVariableType]
                dist = float(row.get("_distance", 0.0))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

                # Filter by max_distance if specified
                if max_distance is not None and dist > max_distance:
                    continue

                # Extract metadata (everything except id, vector, _distance)
                # Type annotations for dict comprehension to satisfy type checker
                metadata: dict[str, Any] = {
                    k: v
                    for k, v in row.items()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                    if k not in ("id", "vector", "_distance")
                }

                query_results.append(
                    QueryResult(
                        id=str(row["id"]),  # pyright: ignore[reportUnknownArgumentType]
                        score=dist,
                        metadata=metadata,
                    )
                )

            logger.debug(
                f"Query returned {len(query_results)} results from namespace {namespace}"
            )
            return query_results

        except Exception:
            logger.exception(f"Failed to query namespace {namespace}")
            raise

    def _build_where_clause(self, filters: dict[str, Any]) -> str | None:
        """
        Convert a filter dict to SQL WHERE clause syntax.

        Args:
            filters: Dictionary of attribute -> value filters

        Returns:
            SQL WHERE clause string or None if no filters
        """
        if not filters:
            return None

        conditions: list[str] = []
        for key, value in filters.items():
            # Handle string values with proper quoting
            if isinstance(value, str):
                # Escape single quotes in the value
                escaped_value = value.replace("'", "''")
                conditions.append(f"{key} = '{escaped_value}'")
            elif isinstance(value, bool):
                conditions.append(f"{key} = {str(value).lower()}")
            elif value is None:
                conditions.append(f"{key} IS NULL")
            else:
                conditions.append(f"{key} = {value}")

        return " AND ".join(conditions) if conditions else None

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from LanceDB.

        Args:
            namespace: The namespace (table) containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        table = self._get_table(namespace)
        if table is None:
            logger.debug(f"Table {namespace} does not exist, nothing to delete")
            return

        try:
            # Build IN clause with properly escaped IDs
            escaped_ids = [f"'{id.replace(chr(39), chr(39) + chr(39))}'" for id in ids]
            in_clause = ", ".join(escaped_ids)
            table.delete(f"id IN ({in_clause})")
            logger.debug(f"Deleted {len(ids)} vectors from namespace {namespace}")
        except Exception:
            logger.exception(
                f"Failed to delete {len(ids)} vectors from namespace {namespace}"
            )
            raise

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace (table) and all its vectors from LanceDB.

        Args:
            namespace: The namespace (table) to delete
        """
        try:
            if namespace in self._db.table_names():
                self._db.drop_table(namespace)
                logger.debug(f"Deleted namespace {namespace}")
            else:
                logger.debug(f"Namespace {namespace} does not exist, nothing to delete")
        except Exception:
            logger.exception(f"Failed to delete namespace {namespace}")
            raise
