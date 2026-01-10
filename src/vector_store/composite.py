"""
Composite vector store implementation.

This module provides a composite VectorStore that writes to two stores
and reads from primary with fallback to secondary.
"""

import asyncio
import logging
from typing import Any

from . import VectorQueryResult, VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)


class CompositeVectorStore(VectorStore):
    """
    Composite vector store with dual-write and fallback-read.

    Behavior:
    - Writes go to BOTH primary and secondary stores
    - Reads try primary only, fall back to secondary on failure

    Migration Strategy:
    - Primary = source of truth (pgvector)
    - Secondary = target being populated (turbopuffer)
    - Reconciliation job syncs data from primary to secondary via dual-writes
    - Reads use primary, fallback to secondary only on exception (network errors, etc.)
    - After migration completes, remove secondary from config
    """

    primary: VectorStore
    secondary: VectorStore

    def __init__(self, primary: VectorStore, secondary: VectorStore):
        """
        Initialize the composite vector store.

        Args:
            primary: The primary vector store (reads prefer this)
            secondary: The secondary vector store (fallback for reads)
        """
        super().__init__()
        self.primary = primary
        self.secondary = secondary

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        """
        Upsert multiple vectors to both stores.

        Success cases:
        - Primary ✓, Secondary ✓ → Success (fully synced)
        - Primary ✓, Secondary ✗ → Returns partial result (not fully synced)

        Failure cases:
        - Primary ✗, Secondary ✓ → Raises primary exception (weird, shouldn't happen)
        - Primary ✗, Secondary ✗ → Raises primary exception (total failure)

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert

        Returns:
            Result describing primary/secondary outcomes.

        Raises:
            Exception: Primary failed (secondary state doesn't matter)
        """
        if not vectors:
            return VectorUpsertResult(primary_ok=True, secondary_ok=True)

        # Write to both stores concurrently
        primary_task = asyncio.create_task(self.primary.upsert_many(namespace, vectors))
        secondary_task = asyncio.create_task(
            self.secondary.upsert_many(namespace, vectors)
        )

        # Wait for both, gathering exceptions
        results: tuple[
            VectorUpsertResult | BaseException, VectorUpsertResult | BaseException
        ] = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

        primary_result, secondary_result = results

        # Case 1: Both failed → raise primary exception
        if isinstance(primary_result, Exception) and isinstance(
            secondary_result, Exception
        ):
            logger.error(
                f"Both primary and secondary upsert failed for namespace {namespace}. Primary: {primary_result}, Secondary: {secondary_result}"
            )
            raise primary_result

        # Case 2: Primary failed, secondary succeeded → raise primary exception (weird case)
        if isinstance(primary_result, Exception):
            logger.error(
                f"Primary upsert failed but secondary succeeded for namespace {namespace}: {primary_result}"
            )
            raise primary_result

        # Case 3: Primary succeeded, secondary failed → return partial result
        if isinstance(secondary_result, Exception):
            logger.warning(
                f"Primary upsert succeeded but secondary failed for namespace {namespace}: {secondary_result}"
            )
            return VectorUpsertResult(
                primary_ok=True,
                secondary_ok=False,
                secondary_error=secondary_result,
            )

        # Case 4: Both succeeded → log success
        logger.debug(
            f"Dual-write upserted {len(vectors)} vectors to namespace {namespace}"
        )
        return VectorUpsertResult(primary_ok=True, secondary_ok=True)

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
        Query for similar vectors, trying primary first then falling back to secondary on failure.

        Primary is the source of truth. Secondary is only used if primary query raises an
        exception (network errors, timeouts, etc.). If primary returns empty results ([]),
        that's considered success and we don't query secondary.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)

        Raises:
            Exception: If both primary and secondary queries fail
        """
        try:
            results = await self.primary.query(
                namespace,
                embedding,
                top_k=top_k,
                filters=filters,
                max_distance=max_distance,
            )
            logger.debug(
                f"Primary query returned {len(results)} results from namespace {namespace}"
            )
            return results
        except Exception as primary_error:
            logger.warning(
                f"Primary query failed for namespace {namespace}: {primary_error}, attempting fallback to secondary"
            )
            try:
                results = await self.secondary.query(
                    namespace,
                    embedding,
                    top_k=top_k,
                    filters=filters,
                    max_distance=max_distance,
                )
                logger.warning(
                    f"Secondary query returned {len(results)} results from namespace {namespace}"
                )
                return results
            except Exception as secondary_error:
                logger.error(
                    f"Both primary and secondary queries failed for namespace {namespace}. Primary: {primary_error}, Secondary: {secondary_error}"
                )
                raise primary_error from secondary_error

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete vectors from both stores.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        # Delete from both stores concurrently
        primary_task = asyncio.create_task(self.primary.delete_many(namespace, ids))
        secondary_task = asyncio.create_task(self.secondary.delete_many(namespace, ids))

        results: tuple[
            None | BaseException, None | BaseException
        ] = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

        primary_result, secondary_result = results

        # Primary failure is critical - raise it
        if isinstance(primary_result, Exception):
            logger.error(
                f"Primary vector store delete failed for namespace {namespace}: {primary_result}"
            )
            raise primary_result

        if isinstance(secondary_result, Exception):
            logger.warning(
                f"Secondary vector store delete failed for namespace {namespace}: {secondary_result}"
            )

        logger.debug(
            f"Dual-delete removed {len(ids)} vectors from namespace {namespace}"
        )

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace from both stores.

        Args:
            namespace: The namespace to delete
        """
        # Delete from both stores concurrently
        primary_task = asyncio.create_task(self.primary.delete_namespace(namespace))
        secondary_task = asyncio.create_task(self.secondary.delete_namespace(namespace))

        results: tuple[
            None | BaseException, None | BaseException
        ] = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

        primary_result, secondary_result = results

        # Primary failure is critical - raise it
        if isinstance(primary_result, Exception):
            logger.error(
                f"Primary vector store namespace delete failed for {namespace}: {primary_result}"
            )
            raise primary_result

        if isinstance(secondary_result, Exception):
            logger.warning(
                f"Secondary vector store namespace delete failed for {namespace}: {secondary_result}"
            )

        logger.debug(f"Dual-delete removed namespace {namespace}")

    async def close(self) -> None:
        """Close both primary and secondary vector stores."""
        await asyncio.gather(
            self.primary.close(),
            self.secondary.close(),
            return_exceptions=True,
        )
        logger.debug("Composite vector store closed")
