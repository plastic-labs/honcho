"""
Milvus vector store implementation.

This module provides a Milvus-based implementation of the VectorStore interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeVar, cast

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from src.config import settings
from src.exceptions import VectorStoreError

from . import VectorQueryResult, VectorRecord, VectorStore

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

ID_FIELD = "id"
VECTOR_FIELD = "vector"
METADATA_FIELD = "metadata"
DISTANCE_METRIC = "COSINE"
MAX_VARCHAR_LENGTH = 65535
MAX_COLLECTION_NAME_LENGTH = 255

STANDARD_METADATA_FIELDS: tuple[str, ...] = (
    "workspace_name",
    "observer",
    "observed",
    "session_name",
    "peer_name",
    "level",
)
RESERVED_FIELDS = {ID_FIELD, VECTOR_FIELD, METADATA_FIELD}
_VALID_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_COLLECTION_CHARS = re.compile(r"[^A-Za-z0-9_]")


class MilvusVectorStore(VectorStore):
    """
    Milvus implementation of the VectorStore interface.

    Honcho's logical namespaces are mapped to deterministic Milvus collection
    names because Milvus collection names are stricter than Honcho namespace
    strings.
    """

    client: MilvusClient

    def __init__(self) -> None:
        """Initialize the Milvus vector store."""
        super().__init__()

        client_kwargs: dict[str, Any] = {"uri": settings.VECTOR_STORE.MILVUS_URI}
        if settings.VECTOR_STORE.MILVUS_TOKEN:
            client_kwargs["token"] = settings.VECTOR_STORE.MILVUS_TOKEN
        if settings.VECTOR_STORE.MILVUS_DB_NAME:
            client_kwargs["db_name"] = settings.VECTOR_STORE.MILVUS_DB_NAME

        self.client = MilvusClient(**client_kwargs)
        self._collection_locks: dict[str, asyncio.Lock] = {}
        self._collection_locks_guard: asyncio.Lock = asyncio.Lock()

    async def _run_client_call(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Run a synchronous MilvusClient call off the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _collection_name(self, namespace: str) -> str:
        """Map a Honcho namespace to a valid, deterministic Milvus collection."""
        sanitized = _UNSAFE_COLLECTION_CHARS.sub("_", namespace).strip("_")
        if not sanitized:
            sanitized = "namespace"
        if not re.match(r"^[A-Za-z_]", sanitized):
            sanitized = f"ns_{sanitized}"

        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
        suffix = f"_{digest}"
        max_stem_length = MAX_COLLECTION_NAME_LENGTH - len(suffix)
        stem = sanitized[:max_stem_length].rstrip("_") or "namespace"
        if not re.match(r"^[A-Za-z_]", stem):
            stem = f"ns_{stem}"
            stem = stem[:max_stem_length].rstrip("_")

        return f"{stem}{suffix}"

    async def _get_collection_lock(self, collection_name: str) -> asyncio.Lock:
        """Get a per-collection lock for create/validate operations."""
        async with self._collection_locks_guard:
            lock = self._collection_locks.get(collection_name)
            if lock is None:
                lock = asyncio.Lock()
                self._collection_locks[collection_name] = lock
            return lock

    async def _has_collection(self, collection_name: str) -> bool:
        """Return whether a Milvus collection exists."""
        has_collection = cast(Callable[..., bool], self.client.has_collection)
        return await self._run_client_call(
            has_collection, collection_name=collection_name
        )

    async def _get_or_create_collection(self, namespace: str) -> str:
        """
        Get an existing collection or create it for the namespace.

        Existing collections are validated before reuse to catch dimension or
        schema mismatches early.
        """
        collection_name = self._collection_name(namespace)
        lock = await self._get_collection_lock(collection_name)

        async with lock:
            if await self._has_collection(collection_name):
                await self._validate_collection_schema(collection_name)
                return collection_name

            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(
                field_name=ID_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=MAX_VARCHAR_LENGTH,
            )
            schema.add_field(
                field_name=VECTOR_FIELD,
                datatype=DataType.FLOAT_VECTOR,
                dim=settings.EMBEDDING.VECTOR_DIMENSIONS,
            )
            schema.add_field(
                field_name=METADATA_FIELD,
                datatype=DataType.JSON,
                nullable=True,
            )
            for field_name in STANDARD_METADATA_FIELDS:
                schema.add_field(
                    field_name=field_name,
                    datatype=DataType.VARCHAR,
                    max_length=MAX_VARCHAR_LENGTH,
                    nullable=True,
                )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type=DISTANCE_METRIC,
            )

            create_kwargs: dict[str, Any] = {
                "collection_name": collection_name,
                "schema": schema,
                "index_params": index_params,
            }
            if settings.VECTOR_STORE.MILVUS_CONSISTENCY_LEVEL:
                create_kwargs["consistency_level"] = (
                    settings.VECTOR_STORE.MILVUS_CONSISTENCY_LEVEL
                )

            create_collection = cast(Callable[..., None], self.client.create_collection)
            try:
                await self._run_client_call(create_collection, **create_kwargs)
            except MilvusException as exc:
                if await self._has_collection(collection_name):
                    await self._validate_collection_schema(collection_name)
                    return collection_name
                raise VectorStoreError(
                    f"Failed to create Milvus collection {collection_name!r}"
                ) from exc
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to create Milvus collection {collection_name!r}"
                ) from exc

        return collection_name

    async def _describe_collection(self, collection_name: str) -> dict[str, Any]:
        """Describe a collection using the Milvus client."""
        describe_collection = cast(
            Callable[..., dict[str, Any]], self.client.describe_collection
        )
        return await self._run_client_call(
            describe_collection, collection_name=collection_name
        )

    async def _validate_collection_schema(self, collection_name: str) -> None:
        """Validate that an existing collection matches Honcho's schema."""
        description = await self._describe_collection(collection_name)
        fields = cast(list[dict[str, Any]], description.get("fields", []))
        by_name = {str(field.get("name")): field for field in fields}

        id_field = by_name.get(ID_FIELD)
        if id_field is None or not id_field.get("is_primary"):
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} must have primary key"
                + f" field {ID_FIELD!r}"
            )
        if id_field.get("type") != DataType.VARCHAR:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} primary key must be VARCHAR"
            )

        actual_dim = self._extract_vector_dim(description)
        expected_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
        if actual_dim != expected_dim:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} vector dim ({actual_dim})"
                + f" does not match EMBEDDING_VECTOR_DIMENSIONS ({expected_dim})"
            )

    def _extract_vector_dim(self, description: dict[str, Any]) -> int:
        """Extract the vector dimension from a Milvus collection description."""
        fields = cast(list[dict[str, Any]], description.get("fields", []))
        for field in fields:
            if field.get("name") != VECTOR_FIELD:
                continue
            if field.get("type") != DataType.FLOAT_VECTOR:
                raise VectorStoreError(
                    "Milvus collection vector field must be FLOAT_VECTOR"
                )
            params = cast(dict[str, Any], field.get("params") or {})
            dim = params.get("dim")
            if dim is None:
                break
            return int(dim)
        raise VectorStoreError(
            f"Milvus collection exists but has no {VECTOR_FIELD!r} field"
            + " with a declared dimension"
        )

    def _validate_embedding_dim(
        self, namespace: str, vectors: list[VectorRecord]
    ) -> None:
        """Validate record dimensions before sending them to Milvus."""
        expected_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
        for record in vectors:
            if len(record.embedding) != expected_dim:
                raise VectorStoreError(
                    f"Vector {record.id!r} in namespace {namespace!r} has dim"
                    + f" {len(record.embedding)}; expected {expected_dim}"
                )

    def _row_to_dict(self, vector: VectorRecord) -> dict[str, Any]:
        """Convert a VectorRecord to a Milvus row."""
        metadata = dict(vector.metadata or {})
        row: dict[str, Any] = {
            ID_FIELD: vector.id,
            VECTOR_FIELD: [float(value) for value in vector.embedding],
            METADATA_FIELD: metadata,
        }

        for key, value in metadata.items():
            if key in RESERVED_FIELDS or not _VALID_IDENTIFIER_PATTERN.match(key):
                continue
            row[key] = value

        for key in STANDARD_METADATA_FIELDS:
            row[key] = self._metadata_varchar_value(metadata.get(key))

        return row

    def _metadata_varchar_value(self, value: Any) -> str | None:
        """Normalize metadata values for VARCHAR fields."""
        if value is None:
            return None
        return str(value)

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> None:
        """
        Upsert multiple vectors into Milvus.

        Args:
            namespace: The namespace to store the vectors in
            vectors: List of VectorRecord objects to upsert
        """
        if not vectors:
            return

        self._validate_embedding_dim(namespace, vectors)
        collection_name = await self._get_or_create_collection(namespace)
        rows = [self._row_to_dict(vector) for vector in vectors]

        upsert = cast(Callable[..., dict[str, Any]], self.client.upsert)
        try:
            await self._run_client_call(
                upsert,
                collection_name=collection_name,
                data=rows,
            )
            logger.debug(
                "Upserted %s vectors to Milvus namespace %s",
                len(vectors),
                namespace,
            )
        except Exception as exc:
            logger.exception(
                "Failed to upsert %s vectors to Milvus namespace %s",
                len(vectors),
                namespace,
            )
            raise VectorStoreError(
                f"Failed to upsert {len(vectors)} vectors to namespace {namespace}"
            ) from exc

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
        Query for similar vectors in Milvus.

        Args:
            namespace: The namespace to query
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            max_distance: Optional maximum distance threshold (cosine distance)
            include_attributes: Attributes to return with each result. False returns
                no metadata; a list returns only those metadata fields.

        Returns:
            List of VectorQueryResult objects, ordered by similarity.
        """
        collection_name = self._collection_name(namespace)
        if not await self._has_collection(collection_name):
            logger.debug(
                "Milvus namespace %s does not exist, returning empty", namespace
            )
            return []

        expected_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
        if len(embedding) != expected_dim:
            raise VectorStoreError(
                f"Query vector for namespace {namespace!r} has dim {len(embedding)};"
                + f" expected {expected_dim}"
            )

        output_fields = self._output_fields(include_attributes)
        search_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "data": [[float(value) for value in embedding]],
            "filter": self._build_filter_expression(filters) if filters else "",
            "limit": top_k,
            "search_params": {"metric_type": DISTANCE_METRIC},
        }
        if output_fields is not None:
            search_kwargs["output_fields"] = output_fields

        try:
            search = cast(Callable[..., list[list[dict[str, Any]]]], self.client.search)
            batches = await self._run_client_call(search, **search_kwargs)
        except Exception as exc:
            logger.exception("Failed to query Milvus namespace %s", namespace)
            raise VectorStoreError(f"Failed to query namespace {namespace}") from exc

        results: list[VectorQueryResult] = []
        for hit in batches[0] if batches else []:
            cosine_distance = self._hit_cosine_distance(hit)
            if max_distance is not None and cosine_distance > max_distance:
                continue

            entity = cast(dict[str, Any], hit.get("entity") or {})
            vector_id = str(hit.get(ID_FIELD) or entity.get(ID_FIELD))
            results.append(
                VectorQueryResult(
                    id=vector_id,
                    score=cosine_distance,
                    metadata=self._entity_metadata(entity),
                )
            )

        logger.debug(
            "Query returned %s results from Milvus namespace %s",
            len(results),
            namespace,
        )
        return results

    def _hit_cosine_distance(self, hit: dict[str, Any]) -> float:
        """Return Honcho cosine distance from a Milvus search hit."""
        if "distance" in hit:
            return float(hit["distance"])
        if "score" in hit:
            return 1.0 - float(hit["score"])
        return 0.0

    def _output_fields(self, include_attributes: bool | list[str]) -> list[str] | None:
        """Translate Honcho projection settings to Milvus output fields."""
        if include_attributes is True:
            return None
        if include_attributes is False:
            return [ID_FIELD]
        return [field for field in include_attributes if field != ID_FIELD]

    def _entity_metadata(self, entity: dict[str, Any]) -> dict[str, Any]:
        """Convert a Milvus search entity into Honcho metadata."""
        metadata: dict[str, Any] = {}
        raw_metadata = entity.get(METADATA_FIELD)
        if isinstance(raw_metadata, dict):
            metadata.update(cast(dict[str, Any], raw_metadata))

        for key, value in entity.items():
            if key in RESERVED_FIELDS or value is None:
                continue
            metadata[key] = value

        return metadata

    def _build_filter_expression(self, filters: dict[str, Any]) -> str:
        """
        Convert a filter dict to a Milvus boolean expression.

        Supports filter formats:
        - {"key": "value"} -> key == "value"
        - {"key": {"in": ["a", "b"]}} -> key in ["a", "b"]
        - {"key": None} -> key is null
        """
        conditions: list[str] = []
        for key, value in filters.items():
            if not _VALID_IDENTIFIER_PATTERN.match(key):
                raise ValueError(f"Invalid filter key: {key!r}")

            if isinstance(value, dict):
                operators = cast(dict[str, Any], value)
                if set(operators.keys()) != {"in"}:
                    raise ValueError(f"Unsupported filter operator for key: {key!r}")
                in_values = cast(Sequence[Any], operators["in"])
                if not in_values:
                    continue
                formatted = ", ".join(self._format_filter_value(v) for v in in_values)
                conditions.append(f"{key} in [{formatted}]")
            elif value is None:
                conditions.append(f"{key} is null")
            else:
                conditions.append(f"{key} == {self._format_filter_value(value)}")

        return " and ".join(conditions)

    def _format_filter_value(self, value: Any) -> str:
        """Format a scalar value for a Milvus filter expression."""
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int | float):
            return str(value)
        raise ValueError(f"Unsupported Milvus filter value: {value!r}")

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        """
        Delete multiple vectors from Milvus.

        Args:
            namespace: The namespace containing the vectors
            ids: List of vector identifiers to delete
        """
        if not ids:
            return

        collection_name = self._collection_name(namespace)
        if not await self._has_collection(collection_name):
            logger.debug(
                "Milvus namespace %s does not exist, nothing to delete", namespace
            )
            return

        delete = cast(Callable[..., dict[str, int]], self.client.delete)
        try:
            await self._run_client_call(
                delete,
                collection_name=collection_name,
                ids=ids,
            )
            logger.debug(
                "Deleted %s vectors from Milvus namespace %s", len(ids), namespace
            )
        except Exception as exc:
            logger.exception(
                "Failed to delete %s vectors from Milvus namespace %s",
                len(ids),
                namespace,
            )
            raise VectorStoreError(
                f"Failed to delete {len(ids)} vectors from namespace {namespace}"
            ) from exc

    async def delete_namespace(self, namespace: str) -> None:
        """
        Delete an entire namespace from Milvus.

        Args:
            namespace: The namespace to delete
        """
        collection_name = self._collection_name(namespace)
        if not await self._has_collection(collection_name):
            logger.debug(
                "Milvus namespace %s does not exist, nothing to delete", namespace
            )
            return

        drop_collection = cast(Callable[..., None], self.client.drop_collection)
        try:
            await self._run_client_call(
                drop_collection,
                collection_name=collection_name,
            )
            logger.debug("Deleted Milvus namespace %s", namespace)
        except Exception as exc:
            logger.exception("Failed to delete Milvus namespace %s", namespace)
            raise VectorStoreError(f"Failed to delete namespace {namespace}") from exc

    async def close(self) -> None:
        """Close the Milvus client connection and release resources."""
        await self._run_client_call(self.client.close)
        logger.debug("Milvus client closed")

    async def probe_namespace_dim(self, namespace: str) -> int | None:
        """Inspect a Milvus collection's vector field dimension.

        Returns ``None`` only when the collection does not exist. When the
        collection exists but lacks the expected vector field, raises
        ``VectorStoreError`` so the startup validator fails closed.
        """
        collection_name = self._collection_name(namespace)
        if not await self._has_collection(collection_name):
            return None

        description = await self._describe_collection(collection_name)
        return self._extract_vector_dim(description)
