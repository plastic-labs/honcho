"""
LanceDB vector store implementation.

This module provides a LanceDB-based implementation of the VectorStore interface.
"""

import asyncio
import logging
from typing import Any, cast

import lancedb
import pyarrow as pa
from lancedb import AsyncConnection, AsyncTable

from src.config import settings

from . import VectorQueryResult, VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)

# Schema for LanceDB tables
# id: string, vector: fixed_size_list of float32 (1536 dimensions for OpenAI embeddings)
# Additional metadata columns are added dynamically

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false


class LanceDBVectorStore(VectorStore):
    """
    LanceDB implementation of the VectorStore interface.

    Uses LanceDB's async embedded mode for local vector storage.
    Each namespace corresponds to a LanceDB table.
    """

    _db: AsyncConnection | None = None
    _db_path: str
    _db_lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialize the LanceDB vector store."""
        super().__init__()
        self._db_path = settings.VECTOR_STORE.LANCEDB_PATH
        self._db = None
        self._db_lock = asyncio.Lock()

    async def _get_db(self) -> AsyncConnection:
        """Get or create the async database connection (asyncio-safe)."""
        if self._db is not None:
            return self._db

        async with self._db_lock:
            # Double-check after acquiring lock
            if self._db is None:
                self._db = await lancedb.connect_async(self._db_path)
        return self._db

    async def _get_table(self, namespace: str) -> AsyncTable | None:
        """Get a table if it exists, otherwise return None."""
        db = await self._get_db()
        table_names = await db.table_names()
        if namespace in table_names:
            return await db.open_table(namespace)
        return None

    async def _get_or_create_table(
        self,
        namespace: str,
    ) -> AsyncTable:
        """
        Get existing table or create if not exists.

        Args:
            namespace: Table name (namespace)

        Returns:
            LanceDB async table
        """
        db = await self._get_db()
        table_names = await db.table_names()
        if namespace in table_names:
            return await db.open_table(namespace)

        # Create empty table with base schema
        fields: list[pa.Field] = [
            pa.field("id", pa.string()),
            pa.field(
                "vector", pa.list_(pa.float32(), settings.VECTOR_STORE.DIMENSIONS)
            ),
        ]
        fields.extend(self._metadata_fields_for_namespace(namespace))
        schema = pa.schema(fields)
        table = await db.create_table(namespace, schema=schema)  # pyright: ignore[reportUnknownArgumentType]
        return table

    def _metadata_fields_for_namespace(self, namespace: str) -> list[pa.Field]:
        """
        Infer standard metadata columns based on namespace structure.

        Namespaces:
            - Documents: {prefix}.{workspace}.{observer}.{observed}
            - Messages:  {prefix}.{workspace}.messages
        """
        parts = namespace.split(".")
        if len(parts) < 3:
            return []

        if parts[-1] == "messages":
            return [
                pa.field("message_id", pa.string(), nullable=True),
                pa.field("session_name", pa.string(), nullable=True),
                pa.field("peer_name", pa.string(), nullable=True),
            ]

        if len(parts) == 4:
            return [
                pa.field("workspace_name", pa.string(), nullable=True),
                pa.field("observer", pa.string(), nullable=True),
                pa.field("observed", pa.string(), nullable=True),
                pa.field("session_name", pa.string(), nullable=True),
                pa.field("level", pa.string(), nullable=True),
            ]

        return []

    def _row_to_dict(self, vector: VectorRecord) -> dict[str, Any]:
        """Convert a VectorRecord to a dict for LanceDB."""
        row: dict[str, Any] = {
            "id": vector.id,
            "vector": vector.embedding,
        }
        # Add metadata fields
        if vector.metadata:
            reserved_keys = {"id", "vector", "_distance"}
            for key in vector.metadata:
                if key not in reserved_keys:
                    row[key] = vector.metadata[key]
        return row

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        """
        Upsert multiple vectors into LanceDB.

        Args:
            namespace: The namespace (table) to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        if not vectors:
            return VectorUpsertResult(primary_ok=True)

        try:
            rows = [self._row_to_dict(v) for v in vectors]
            table = await self._get_or_create_table(namespace)

            # Use merge_insert for upsert behavior
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )

            logger.debug(f"Upserted {len(vectors)} vectors to namespace {namespace}")
            return VectorUpsertResult(primary_ok=True)
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
    ) -> list[VectorQueryResult]:
        """
        Query for similar vectors in LanceDB.

        Args:
            namespace: The namespace (table) to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)
        """
        table = await self._get_table(namespace)
        if table is None:
            logger.debug(f"Table {namespace} does not exist, returning empty results")
            return []

        try:
            # Build query
            query = table.vector_search(embedding).distance_type("cosine").limit(top_k)

            # Apply filters if provided
            if filters:
                where_clause = self._build_where_clause(filters)
                if where_clause:
                    query = query.where(where_clause)

            # Execute query
            # LanceDB async API returns list of dicts with incomplete type annotations
            results = cast(list[dict[str, Any]], await query.to_list())

            # Convert to VectorQueryResult objects
            query_results: list[VectorQueryResult] = []
            for row in results:
                dist = float(row.get("_distance", 0.0))

                # Filter by max_distance if specified
                if max_distance is not None and dist > max_distance:
                    continue

                # Extract metadata (everything except id, vector, _distance)
                metadata: dict[str, Any] = {
                    k: v
                    for k, v in row.items()
                    if k not in ("id", "vector", "_distance")
                }

                query_results.append(
                    VectorQueryResult(
                        id=str(row["id"]),
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

        table = await self._get_table(namespace)
        if table is None:
            logger.debug(f"Table {namespace} does not exist, nothing to delete")
            return

        try:
            # Build IN clause with properly escaped IDs
            escaped_ids = [f"'{id.replace(chr(39), chr(39) + chr(39))}'" for id in ids]
            in_clause = ", ".join(escaped_ids)
            await table.delete(f"id IN ({in_clause})")
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
            db = await self._get_db()
            table_names = await db.table_names()
            if namespace in table_names:
                await db.drop_table(namespace)
            else:
                logger.debug(f"Namespace {namespace} does not exist, nothing to delete")
        except Exception:
            logger.exception(f"Failed to delete namespace {namespace}")
            raise

    async def close(self) -> None:
        """Close the LanceDB connection and release resources."""
        if self._db is not None:
            # LanceDB AsyncConnection doesn't have an explicit close method,
            # but we clear the reference to allow garbage collection
            self._db = None
            logger.debug("LanceDB connection closed")
