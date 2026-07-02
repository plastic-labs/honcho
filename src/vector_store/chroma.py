"""
ChromaDB vector store implementation.

This module provides a ChromaDB-based implementation of the VectorStore
interface, supporting three deployment modes selected by
VECTOR_STORE_CHROMA_CLIENT_MODE:

- "persistent": local embedded storage (chromadb.PersistentClient)
- "http": self-hosted Chroma server (chromadb.HttpClient)
- "cloud": Chroma Cloud (chromadb.CloudClient)

Chroma's persistent and cloud clients are synchronous, so every client call
is offloaded through asyncio.to_thread to keep the event loop free. One code
path covers all three modes.

Each Honcho namespace maps to one Chroma collection. Honcho namespace strings
({prefix}.{type}.{base64url-hash}) are not valid Chroma collection names
(uppercase characters, arbitrary leading/trailing chars), so the collection
name is a deterministic re-hash: "ns-" + sha256(namespace).hexdigest(). The
original namespace is stored in collection metadata for debuggability.
"""

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING, Any, cast

import httpx

from src.config import settings
from src.exceptions import VectorStoreError

from . import VectorQueryResult, VectorRecord, VectorStore

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# Chroma Cloud caps write batches at 300 records; local SQLite-backed builds
# expose their own limit via client.get_max_batch_size(). Batches are split
# to min of both so one upsert_many call never exceeds either.
CLOUD_MAX_BATCH_SIZE = 300

# Errors that indicate the store is unreachable/unavailable rather than a
# logic error. Chroma's http/cloud clients surface transport failures as
# httpx errors; persistent mode has no transport layer.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of the VectorStore interface.

    Each namespace corresponds to a Chroma collection created with cosine
    distance and no embedding function (Honcho always supplies precomputed
    vectors). Collections are lazy-created on first write; reads and deletes
    against a missing collection are empty results / no-ops.
    """

    _client: "ClientAPI | None"
    _client_lock: asyncio.Lock
    _max_batch_size: int | None

    def __init__(self) -> None:
        """Initialize the ChromaDB vector store."""
        super().__init__()
        self._client = None
        self._client_lock = asyncio.Lock()
        self._max_batch_size = None

    def _create_client(self) -> "ClientAPI":
        """Construct the sync Chroma client for the configured mode.

        Runs inside a worker thread: PersistentClient does disk I/O and the
        http/cloud clients issue a version heartbeat on construction.
        """
        import chromadb

        mode = settings.VECTOR_STORE.CHROMA_CLIENT_MODE
        if mode == "persistent":
            return chromadb.PersistentClient(path=settings.VECTOR_STORE.CHROMA_PATH)
        elif mode == "http":
            return chromadb.HttpClient(
                host=settings.VECTOR_STORE.CHROMA_HOST,
                port=settings.VECTOR_STORE.CHROMA_PORT,
                ssl=settings.VECTOR_STORE.CHROMA_SSL,
            )
        else:
            api_key = settings.VECTOR_STORE.CHROMA_API_KEY
            if not api_key:
                raise ValueError(
                    "VECTOR_STORE_CHROMA_API_KEY must be set for ChromaDB cloud mode"
                )
            return chromadb.CloudClient(
                api_key=api_key,
                tenant=settings.VECTOR_STORE.CHROMA_TENANT,
                database=settings.VECTOR_STORE.CHROMA_DATABASE,
            )

    async def _get_client(self) -> "ClientAPI":
        """Get or create the Chroma client (asyncio-safe)."""
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is None:
                self._client = await asyncio.to_thread(self._create_client)
        return self._client

    async def _get_batch_limit(self, client: "ClientAPI") -> int:
        """Resolve the per-request write batch limit for this client."""
        if self._max_batch_size is None:
            try:
                local_limit = await asyncio.to_thread(client.get_max_batch_size)
            except Exception:
                # Some server versions don't expose the pre-flight limit;
                # fall back to the strictest known cap.
                local_limit = CLOUD_MAX_BATCH_SIZE
            self._max_batch_size = max(1, min(local_limit, CLOUD_MAX_BATCH_SIZE))
        return self._max_batch_size

    @staticmethod
    def _collection_name(namespace: str) -> str:
        """
        Map a Honcho namespace to a valid Chroma collection name.

        Honcho namespaces contain base64url hashes (mixed case, may end in
        '-'/'_') which violate Chroma's naming rules (start/end with a
        lowercase alphanumeric). A sha256 hex digest is deterministic,
        injective, and valid under both OSS and Cloud naming rules.
        """
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        return f"ns-{digest}"

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        """
        Drop metadata entries Chroma cannot store.

        None values are rejected by Chroma Cloud and older servers (the key
        is simply omitted instead — an absent key behaves the same as NULL
        under equality filters). Reserved keys used by the record itself are
        also excluded.
        """
        sanitized = {
            k: v
            for k, v in metadata.items()
            if v is not None and k not in ("id", "embedding")
        }
        return sanitized or None

    def _build_where(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """
        Convert a filter dict to Chroma's `where` format.

        Supports filter formats:
        - {"key": "value"} -> {"key": {"$eq": "value"}}
        - {"key": {"in": [...]}} -> {"key": {"$in": [...]}}
        Multiple filters combine with {"$and": [...]} (Chroma requires at
        least two operands for $and, so a single clause is passed bare).

        Raises:
            ValueError: If a filter value is None (Chroma cannot match null;
                write-side sanitization omits null keys entirely).
        """
        clauses: list[dict[str, Any]] = []
        for key, value in filters.items():
            if isinstance(value, dict) and "in" in value:
                in_values = cast(list[Any], value["in"])
                clauses.append({key: {"$in": list(in_values)}})
            elif value is None:
                raise ValueError(
                    f"ChromaDB backend does not support filtering on null values (key: {key!r})"
                )
            else:
                clauses.append({key: {"$eq": value}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    async def _get_collection(self, namespace: str) -> "Collection | None":
        """Get a collection if it exists, otherwise return None."""
        from chromadb.errors import NotFoundError

        client = await self._get_client()
        try:
            return await asyncio.to_thread(
                client.get_collection, self._collection_name(namespace)
            )
        except NotFoundError:
            return None

    async def _get_or_create_collection(self, namespace: str) -> "Collection":
        """Get or create the collection backing a namespace.

        Created with cosine distance and no embedding function — Honcho
        always supplies precomputed vectors. `space` is the only index knob
        that carries across single-node (HNSW) and Cloud (SPANN).
        """
        client = await self._get_client()
        return await asyncio.to_thread(
            client.get_or_create_collection,
            name=self._collection_name(namespace),
            configuration=cast(Any, {"hnsw": {"space": "cosine"}}),
            embedding_function=None,
            metadata={"honcho_namespace": namespace},
        )

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> None:
        """
        Upsert multiple vectors into ChromaDB.

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        if not vectors:
            return

        try:
            collection = await self._get_or_create_collection(namespace)
            client = await self._get_client()
            batch_limit = await self._get_batch_limit(client)

            for start in range(0, len(vectors), batch_limit):
                batch = vectors[start : start + batch_limit]
                await asyncio.to_thread(
                    collection.upsert,
                    ids=[v.id for v in batch],
                    embeddings=cast(Any, [v.embedding for v in batch]),
                    metadatas=cast(
                        Any, [self._sanitize_metadata(v.metadata) for v in batch]
                    ),
                )

            logger.debug(f"Upserted {len(vectors)} vectors to namespace {namespace}")
        except _TRANSIENT_ERRORS as exc:
            logger.warning(
                "ChromaDB unavailable for upsert to namespace %s: %s",
                namespace,
                exc,
            )
            raise VectorStoreError(
                f"ChromaDB unavailable for upsert to namespace {namespace}"
            ) from exc
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
        include_attributes: bool | list[str] = True,
    ) -> list[VectorQueryResult]:
        """
        Query for similar vectors in ChromaDB.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)
            include_attributes: Attributes to return with each result. False
                skips metadata retrieval entirely; a list projects to those
                fields (Chroma has no per-key projection, so the trim happens
                client-side).

        Returns:
            List of VectorQueryResult objects, ordered by similarity (most similar first)
        """
        try:
            collection = await self._get_collection(namespace)
            if collection is None:
                logger.debug(
                    f"Collection for namespace {namespace} does not exist, returning empty results"
                )
                return []

            where = self._build_where(filters) if filters else None
            include: list[str] = ["distances"]
            if include_attributes is not False:
                include.append("metadatas")

            query_kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": include,
            }
            if where is not None:
                query_kwargs["where"] = where

            response = cast(
                dict[str, Any],
                cast(
                    object,
                    await asyncio.to_thread(collection.query, **query_kwargs),
                ),
            )

            # Chroma returns column-major, per-query-embedding lists; we send
            # exactly one query embedding, so index 0 everywhere.
            ids: list[str] = cast(list[str], (response.get("ids") or [[]])[0])
            distances: list[float] = cast(
                list[float], (response.get("distances") or [[]])[0]
            )
            metadatas_col = response.get("metadatas")
            metadatas: list[dict[str, Any] | None] = (
                metadatas_col[0] if metadatas_col else [None] * len(ids)
            )

            query_results: list[VectorQueryResult] = []
            for row_id, dist, metadata in zip(ids, distances, metadatas, strict=True):
                dist = float(dist)
                if max_distance is not None and dist > max_distance:
                    continue

                row_metadata: dict[str, Any] = dict(metadata) if metadata else {}
                if isinstance(include_attributes, list):
                    row_metadata = {
                        k: v for k, v in row_metadata.items() if k in include_attributes
                    }

                query_results.append(
                    VectorQueryResult(
                        id=str(row_id),
                        score=dist,
                        metadata=row_metadata,
                    )
                )

            logger.debug(
                f"Query returned {len(query_results)} results from namespace {namespace}"
            )
            return query_results

        except _TRANSIENT_ERRORS as exc:
            # Store unavailable — degrade to empty results, matching the
            # Turbopuffer backend's behavior on 5xx.
            logger.warning(
                "ChromaDB unavailable for query on namespace %s, returning empty results: %s",
                namespace,
                exc,
            )
            return []
        except Exception:
            logger.exception(f"Failed to query namespace {namespace}")
            raise

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from ChromaDB.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        try:
            collection = await self._get_collection(namespace)
            if collection is None:
                logger.debug(
                    f"Collection for namespace {namespace} does not exist, nothing to delete"
                )
                return

            client = await self._get_client()
            batch_limit = await self._get_batch_limit(client)
            for start in range(0, len(ids), batch_limit):
                batch = ids[start : start + batch_limit]
                await asyncio.to_thread(collection.delete, ids=batch)
            logger.debug(f"Deleted {len(ids)} vectors from namespace {namespace}")
        except _TRANSIENT_ERRORS as exc:
            logger.warning(
                "ChromaDB unavailable for delete from namespace %s: %s",
                namespace,
                exc,
            )
            raise VectorStoreError(
                f"ChromaDB unavailable while deleting vectors in namespace {namespace}"
            ) from exc
        except Exception:
            logger.exception(
                f"Failed to delete {len(ids)} vectors from namespace {namespace}"
            )
            raise

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace (collection) and all its vectors from ChromaDB.

        Args:
            namespace: The namespace to delete
        """
        from chromadb.errors import NotFoundError

        try:
            client = await self._get_client()
            await asyncio.to_thread(
                client.delete_collection, self._collection_name(namespace)
            )
            logger.debug(f"Deleted namespace {namespace}")
        except NotFoundError:
            logger.debug(f"Namespace {namespace} does not exist, nothing to delete")
        except Exception:
            logger.exception(f"Failed to delete namespace {namespace}")
            raise

    async def close(self) -> None:
        """Close the ChromaDB client and release resources."""
        if self._client is not None:
            client = self._client
            self._client = None
            self._max_batch_size = None
            # Refcounted close() landed in chromadb 1.5.2; guard for safety
            # against client classes that don't expose it.
            close_method = getattr(client, "close", None)
            if callable(close_method):
                await asyncio.to_thread(close_method)
            logger.debug("ChromaDB client closed")

    async def probe_namespace_dim(self, namespace: str) -> int | None:
        """Recover the vector dimension of an existing Chroma collection.

        Chroma locks a collection's dimensionality on first write but does
        not expose it reliably through the collection model, so this peeks
        one stored embedding. Returns ``None`` when the collection does not
        exist (lazy-create model) or exists but has no records yet — in
        both cases no dimension has been locked, which the startup
        validator treats as "nothing to check".
        """
        collection = await self._get_collection(namespace)
        if collection is None:
            return None

        peek = cast(
            dict[str, Any],
            cast(
                object,
                await asyncio.to_thread(
                    collection.get, limit=1, include=cast(Any, ["embeddings"])
                ),
            ),
        )
        embeddings = peek.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None

        first = embeddings[0]
        try:
            return int(len(first))
        except TypeError as exc:
            raise VectorStoreError(
                f"ChromaDB collection for namespace {namespace!r} exists but"
                + " returned an unreadable embedding; cannot probe dim."
            ) from exc
