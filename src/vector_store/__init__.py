"""
Vector store abstraction layer for Honcho.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.config import settings
from src.vector_store.lancedb import LanceDBVectorStore
from src.vector_store.turbopuffer import TurbopufferVectorStore


@dataclass
class VectorRecord:
    """A single vector record to be stored in the vector store."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """A single result from a vector query."""

    id: str
    score: float  # Distance/similarity score (lower = more similar for cosine distance)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    All vector operations are namespace-scoped. Namespaces map to:
    - Document embeddings: {prefix}-{workspace}-{observer}-{observed} (per collection)
    - Message embeddings: {prefix}-{workspace}-messages (per workspace)
    """

    namespace_prefix: str

    def __init__(self):
        """
        Initialize the vector store.
        """
        self.namespace_prefix = settings.VECTOR_STORE.NAMESPACE

    # === Namespace helpers ===
    def get_document_namespace(
        self, workspace_name: str, observer: str, observed: str
    ) -> str:
        """
        Get the namespace for document embeddings (per collection).

        Args:
            workspace_name: Name of the workspace
            observer: Name of the observing peer
            observed: Name of the observed peer

        Returns:
            Namespace string in format: {prefix}_{workspace}_{observer}_{observed}
        """
        return f"{self.namespace_prefix}_{workspace_name}_{observer}_{observed}"

    def get_message_namespace(self, workspace_name: str) -> str:
        """
        Get the namespace for message embeddings (per workspace).

        Args:
            workspace_name: Name of the workspace

        Returns:
            Namespace string in format: {prefix}_{workspace}_messages
        """
        return f"{self.namespace_prefix}_{workspace_name}_messages"

    # === Core operations ===
    @abstractmethod
    async def upsert(
        self,
        namespace: str,
        vector: VectorRecord,
    ) -> None:
        """
        Upsert a single vector into the store.

        Args:
            namespace: The namespace to store the vector in
            id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        pass

    @abstractmethod
    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> None:
        """
        Upsert multiple vectors into the store.

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        pass

    @abstractmethod
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
        Query for similar vectors.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of QueryResult objects, ordered by similarity (most similar first)
        """
        pass

    @abstractmethod
    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from the store.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        pass

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace and all its vectors.

        Args:
            namespace: The namespace to delete
        """
        pass


# Singleton instance
_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Get the configured vector store instance (singleton).

    Returns:
        The vector store instance based on configuration.

    Raises:
        ValueError: If the configured vector store type is invalid.
    """
    global _vector_store_instance

    if _vector_store_instance is not None:
        return _vector_store_instance

    store_type = settings.VECTOR_STORE.TYPE

    if store_type == "turbopuffer":
        _vector_store_instance = TurbopufferVectorStore()
    elif store_type == "lancedb":
        _vector_store_instance = LanceDBVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

    return _vector_store_instance


__all__ = [
    "VectorStore",
    "VectorRecord",
    "QueryResult",
    "get_vector_store",
]
