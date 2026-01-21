"""
Vector store abstraction layer for Honcho.
"""

import base64
import hashlib
from abc import ABC, abstractmethod
from functools import cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import settings


def _hash_namespace_components(*parts: str) -> str:
    """
    Hash namespace components to create a fixed-length, valid namespace suffix.

    Turbopuffer requires namespaces to match [A-Za-z0-9-_.]{1,128}.
    This function hashes the variable parts (workspace, observer, observed)
    to ensure the namespace fits within length limits and uses only valid chars.

    Returns:
        A 43-character base64url-encoded SHA-256 hash (no padding).
        (SHA-256 = 32 bytes, base64 = ceil(32 * 4 / 3) = 43 chars without padding)
    """
    combined = ".".join(parts)
    hash_bytes = hashlib.sha256(combined.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hash_bytes).decode("ascii").rstrip("=")


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
    )

    ok: bool


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    All vector operations are namespace-scoped. Namespaces are generated via
    get_vector_namespace() which hashes workspace/peer names to ensure:
    - Total length fits within Turbopuffer's 128 char limit
    - Only valid characters [A-Za-z0-9-_.] are used

    Namespace format: {prefix}.{type}.{hash}
    - Document embeddings: {prefix}.doc.{hash}
    - Message embeddings: {prefix}.msg.{hash}
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
            - document: {prefix}.doc.{hash}
            - message: {prefix}.msg.{hash}
            where hash is derived from the workspace/peer names.
        """
        if namespace_type == "document":
            if observer is None or observed is None:
                raise ValueError(
                    "observer and observed are required for document namespaces"
                )
            hash_suffix = _hash_namespace_components(workspace_name, observer, observed)
            return f"{self.namespace_prefix}.doc.{hash_suffix}"
        elif namespace_type == "message":
            hash_suffix = _hash_namespace_components(workspace_name)
            return f"{self.namespace_prefix}.msg.{hash_suffix}"

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
from src.vector_store.lancedb import LanceDBVectorStore  # noqa: E402
from src.vector_store.turbopuffer import TurbopufferVectorStore  # noqa: E402
from src.vector_store.utils import upsert_with_retry  # noqa: E402


def _create_store_by_type(store_type: str) -> VectorStore:
    """Create a vector store instance by type name."""
    if store_type == "turbopuffer":
        return TurbopufferVectorStore()
    elif store_type == "lancedb":
        return LanceDBVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


@cache
def get_external_vector_store() -> VectorStore | None:
    """
    Get the configured external vector store instance (singleton).

    Returns None if TYPE='pgvector' since pgvector operations happen via ORM directly.
    External vector stores include Turbopuffer and LanceDB.

    Returns:
        The external vector store instance, or None if using pgvector (ORM handles it).

    Raises:
        ValueError: If the configured vector store type is invalid.
    """
    if settings.VECTOR_STORE.TYPE == "pgvector":
        return None
    return _create_store_by_type(settings.VECTOR_STORE.TYPE)


async def close_external_vector_store() -> None:
    """
    Close the external vector store and release resources.

    Call this during application shutdown to cleanly close connections.
    After calling this, you must call get_external_vector_store.cache_clear() if you
    want to create a new instance.
    """
    # Check if an instance was ever created
    if (
        get_external_vector_store.cache_info().hits > 0
        or get_external_vector_store.cache_info().misses > 0
    ):
        store = get_external_vector_store()
        if store is not None:
            await store.close()
        get_external_vector_store.cache_clear()


__all__ = [
    "VectorStore",
    "VectorRecord",
    "VectorQueryResult",
    "VectorUpsertResult",
    "get_external_vector_store",
    "close_external_vector_store",
    "upsert_with_retry",
    "_hash_namespace_components",
]
