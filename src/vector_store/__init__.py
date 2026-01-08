"""
Vector store abstraction layer for Honcho.
"""

from abc import ABC, abstractmethod
from functools import cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import settings


class VectorRecord(BaseModel):
    """A single vector record to be stored in the vector store."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorQueryResult(BaseModel):
    """A single result from a vector query."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    id: str
    score: float  # Distance/similarity score (lower = more similar for cosine distance)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorUpsertResult(BaseModel):
    """Result for a vector upsert operation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    primary_ok: bool
    secondary_ok: bool | None = None
    secondary_error: Exception | None = None


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    All vector operations are namespace-scoped. Namespaces map to:
    - Document embeddings: {prefix}.{workspace}.{observer}.{observed} (per collection)
    - Message embeddings: {prefix}.{workspace}.messages (per workspace)

    Note: Period (.) is used as the delimiter since vector stores (Turbopuffer, LanceDB)
    only allow [A-Za-z0-9-_.] in namespace names, and period is not allowed in
    workspace/peer IDs (which only allow [A-Za-z0-9_-]).
    """

    namespace_prefix: str

    def __init__(self):
        """
        Initialize the vector store.
        """
        self.namespace_prefix = settings.VECTOR_STORE.NAMESPACE

    # === Namespace helpers ===
    def get_vector_namespace(
        self,
        namespace_type: Literal["document", "message"],
        workspace_name: str,
        observer: str | None = None,
        observed: str | None = None,
    ) -> str:
        """
        Get the namespace for document or message embeddings.

        Args:
            namespace_type: "document" or "message"
            workspace_name: Name of the workspace
            observer: Name of the observing peer (document only)
            observed: Name of the observed peer (document only)

        Returns:
            Namespace string in format:
            - document: {prefix}.{workspace}.{observer}.{observed}
            - message: {prefix}.{workspace}.messages
        """
        if namespace_type == "document":
            if observer is None or observed is None:
                raise ValueError(
                    "observer and observed are required for document namespaces"
                )
            return f"{self.namespace_prefix}.{workspace_name}.{observer}.{observed}"
        if namespace_type == "message":
            return f"{self.namespace_prefix}.{workspace_name}.messages"

    # === Core operations ===
    @abstractmethod
    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        """
        Upsert multiple vectors into the store.

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert

        Returns:
            Result describing primary/secondary outcomes.
        """
        ...

    @abstractmethod
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
        Query for similar vectors.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)
        """
        ...

    @abstractmethod
    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from the store.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        ...

    @abstractmethod
    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace and all its vectors.

        Args:
            namespace: The namespace to delete
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """
        Close any open connections and release resources.

        Subclasses should override this if they maintain persistent connections.
        """
        ...


# Import implementations after base classes are defined to avoid circular imports
from src.vector_store.composite import CompositeVectorStore  # noqa: E402
from src.vector_store.lancedb import LanceDBVectorStore  # noqa: E402
from src.vector_store.pgvector import PgVectorStore  # noqa: E402
from src.vector_store.turbopuffer import TurbopufferVectorStore  # noqa: E402


def _create_store_by_type(store_type: str) -> VectorStore:
    """Create a vector store instance by type name."""
    if store_type == "turbopuffer":
        return TurbopufferVectorStore()
    elif store_type == "lancedb":
        return LanceDBVectorStore()
    elif store_type == "pgvector":
        return PgVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def _create_vector_store() -> VectorStore:
    """
    Create a new vector store instance based on configuration.

    If SECONDARY_TYPE is set, returns a CompositeVectorStore that:
    - Writes to both primary and secondary stores
    - Reads from primary only, falls back to secondary on failure

    Returns:
        The vector store instance based on configuration.

    Raises:
        ValueError: If the configured vector store type is invalid.
    """
    primary = _create_store_by_type(settings.VECTOR_STORE.PRIMARY_TYPE)

    if settings.VECTOR_STORE.SECONDARY_TYPE:
        secondary = _create_store_by_type(settings.VECTOR_STORE.SECONDARY_TYPE)
        return CompositeVectorStore(primary=primary, secondary=secondary)

    return primary


@cache
def get_vector_store() -> VectorStore:
    """
    Get the configured vector store instance (singleton).

    Uses functools.cache to ensure only one instance is created per process.
    This is asyncio-safe since there are no await points in the creation path.

    Returns:
        The vector store instance based on configuration.

    Raises:
        ValueError: If the configured vector store type is invalid.
    """
    return _create_vector_store()


async def close_vector_store() -> None:
    """
    Close the vector store and release resources.

    Call this during application shutdown to cleanly close connections.
    After calling this, you must call get_vector_store.cache_clear() if you
    want to create a new instance.
    """
    # Check if an instance was ever created
    if (
        get_vector_store.cache_info().hits > 0
        or get_vector_store.cache_info().misses > 0
    ):
        store = get_vector_store()
        await store.close()
        get_vector_store.cache_clear()


__all__ = [
    "VectorStore",
    "VectorRecord",
    "VectorQueryResult",
    "VectorUpsertResult",
    "get_vector_store",
    "close_vector_store",
]
