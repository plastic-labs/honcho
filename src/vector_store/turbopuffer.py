"""
Turbopuffer vector store implementation.

This module provides a Turbopuffer-based implementation of the VectorStore interface.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

from turbopuffer import AsyncTurbopuffer, NotFoundError
from turbopuffer.lib.namespace import AsyncNamespace
from turbopuffer.types import Filter

from src.config import settings

from . import VectorQueryResult, VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)

# Type aliases for Turbopuffer's filter formats
EqFilter = tuple[str, Literal["Eq"], Any]
InFilter = tuple[str, Literal["In"], Sequence[Any]]
AndFilter = tuple[Literal["And"], Sequence[Filter]]

DISTANCE_METRIC = "cosine_distance"


class TurbopufferVectorStore(VectorStore):
    """
    Turbopuffer implementation of the VectorStore interface.

    Each namespace corresponds to either:
    - A document collection: {prefix}.{workspace}.{observer}.{observed}
    - A workspace's message embeddings: {prefix}.{workspace}.messages
    """

    tpuf: AsyncTurbopuffer

    def __init__(self):
        """
        Initialize the Turbopuffer vector store.
        """
        super().__init__()

        # Configure Turbopuffer client
        api_key = settings.VECTOR_STORE.TURBOPUFFER_API_KEY
        if not api_key:
            raise ValueError(
                "VECTOR_STORE_TURBOPUFFER_API_KEY must be set for Turbopuffer vector store"
            )

        # Initialize the async Turbopuffer client
        # Region can be configured via VECTOR_STORE_TURBOPUFFER_REGION or TURBOPUFFER_REGION env var
        region = settings.VECTOR_STORE.TURBOPUFFER_REGION or "gcp-us-east4"
        self.tpuf = AsyncTurbopuffer(api_key=api_key, region=region)

    def _get_namespace(self, namespace: str) -> AsyncNamespace:
        """Get a Turbopuffer namespace object."""
        return self.tpuf.namespace(namespace)

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        """
        Upsert multiple vectors into Turbopuffer.

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        if not vectors:
            return VectorUpsertResult(ok=True)

        ns = self._get_namespace(namespace)

        rows: list[dict[str, Any]] = [
            {
                "id": v.id,
                "vector": v.embedding,
                **(v.metadata or {}),
            }
            for v in vectors
        ]

        try:
            await ns.write(
                upsert_rows=rows,
                distance_metric=DISTANCE_METRIC,
            )
            return VectorUpsertResult(ok=True)
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
        Query for similar vectors in Turbopuffer.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)
        """
        ns = self._get_namespace(namespace)

        try:
            # Build filter conditions for Turbopuffer
            filter_condition = self._build_filters(filters) if filters else None

            # Query using rank_by for vector similarity
            # rank_by must be a tuple: (attribute, "ANN", vector)
            rank_by: tuple[str, Literal["ANN"], Sequence[float]] = (
                "vector",
                "ANN",
                embedding,
            )

            # Only pass filters if we have them (avoid passing None)
            query_kwargs: dict[str, Any] = {
                "rank_by": rank_by,
                "top_k": top_k,
                "distance_metric": DISTANCE_METRIC,
                "include_attributes": True,
            }
            if filter_condition is not None:
                query_kwargs["filters"] = filter_condition

            response = await ns.query(**query_kwargs)

            query_results: list[VectorQueryResult] = []
            for row in response.rows or []:
                # Distance is accessed via row["$dist"]
                dist: float = float(row["$dist"]) if "$dist" in row else 0.0
                # Filter by max_distance if specified
                if max_distance is not None and dist > max_distance:
                    continue

                # Extract attributes from model_extra (excludes id, vector, $dist)
                row_metadata: dict[str, Any] = {}
                if row.model_extra:
                    # Filter out internal fields like $dist
                    row_metadata = {
                        k: v
                        for k, v in row.model_extra.items()
                        if not k.startswith("$")
                    }

                query_results.append(
                    VectorQueryResult(
                        id=str(row.id),
                        score=dist,
                        metadata=row_metadata,
                    )
                )

            logger.debug(
                f"Query returned {len(query_results)} results from namespace {namespace}"
            )
            return query_results

        except NotFoundError:
            # Namespace doesn't exist yet - no vectors have been written
            # Return empty results (same behavior as LanceDB for missing tables)
            logger.debug(
                f"Namespace {namespace} does not exist, returning empty results"
            )
            return []

        except Exception:
            logger.exception(f"Failed to query namespace {namespace}")
            raise

    def _build_filters(self, filters: dict[str, Any]) -> Filter | None:
        """
        Convert a filter dict to Turbopuffer filter format.

        Turbopuffer uses tuples like (attribute, "Eq", value) for filters,
        (attribute, "In", [values]) for membership filters,
        and ("And", [filters]) for combining multiple filters.

        Supports filter formats:
        - {"key": "value"} -> ("key", "Eq", "value")
        - {"key": {"in": ["a", "b"]}} -> ("key", "In", ["a", "b"])

        Args:
            filters: Dictionary of attribute -> value filters

        Returns:
            Turbopuffer Filter or None if no filters
        """
        if not filters:
            return None

        filter_list: list[EqFilter | InFilter] = []
        for key, value in filters.items():
            # Check if value is a dict with "in" operator
            if isinstance(value, dict) and "in" in value:
                # Membership filter using "In" operator
                in_values = cast(Sequence[Any], value["in"])
                filter_list.append((key, "In", in_values))
            else:
                # Simple equality filter using "Eq" operator
                filter_list.append((key, "Eq", cast(Any, value)))

        if not filter_list:
            return None

        if len(filter_list) == 1:
            return filter_list[0]

        # Combine multiple filters with AND
        and_filter: AndFilter = ("And", filter_list)
        return and_filter

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from Turbopuffer.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        ns = self._get_namespace(namespace)

        try:
            await ns.write(deletes=ids)
        except NotFoundError:
            # Namespace doesn't exist - nothing to delete
            logger.debug(f"Namespace {namespace} does not exist, nothing to delete")
        except Exception:
            logger.exception(
                f"Failed to delete {len(ids)} vectors from namespace {namespace}"
            )
            raise

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace and all its vectors from Turbopuffer.

        Args:
            namespace: The namespace to delete
        """
        ns = self._get_namespace(namespace)

        try:
            await ns.delete_all()
            logger.debug(f"Deleted all vectors from namespace {namespace}")
        except NotFoundError:
            # Namespace doesn't exist - nothing to delete
            logger.debug(f"Namespace {namespace} does not exist, nothing to delete")
        except Exception:
            logger.exception(f"Failed to delete namespace {namespace}")
            raise

    async def close(self) -> None:
        """Close the Turbopuffer client and release resources."""
        await self.tpuf.close()
        logger.debug("Turbopuffer client closed")
