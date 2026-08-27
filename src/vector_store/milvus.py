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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
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


def _uses_lite_3_cosine_distance(uri: str) -> bool:
    """Return whether the client uses Milvus Lite 3.0 distance semantics."""
    if "://" in uri:
        return False
    try:
        return package_version("milvus-lite") in {"3.0", "3.0.0"}
    except PackageNotFoundError:
        return False


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

        uri = settings.VECTOR_STORE.MILVUS_URI
        if "://" not in uri:
            try:
                package_version("milvus-lite")
            except PackageNotFoundError as exc:
                raise RuntimeError(
                    "A local VECTOR_STORE_MILVUS_URI requires Honcho's "
                    + "'milvus-lite' extra (`uv sync --extra milvus-lite`); use the "
                    + "'milvus' extra only with Milvus Server or Zilliz Cloud"
                ) from exc

        client_kwargs: dict[str, Any] = {"uri": uri}
        if settings.VECTOR_STORE.MILVUS_TOKEN:
            client_kwargs["token"] = settings.VECTOR_STORE.MILVUS_TOKEN
        if settings.VECTOR_STORE.MILVUS_DB_NAME:
            client_kwargs["db_name"] = settings.VECTOR_STORE.MILVUS_DB_NAME

        try:
            self.client = MilvusClient(**client_kwargs)
        except MilvusException as exc:
            raise VectorStoreError(
                "Failed to initialize Milvus client for the configured URI"
            ) from exc
        self._collection_locks: dict[str, asyncio.Lock] = {}
        self._collection_locks_guard: asyncio.Lock = asyncio.Lock()
        self._validated_collections: set[str] = set()

    async def _run_client_call(
        self,
        operation: str,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Run a Milvus call off the event loop and normalize SDK failures."""
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except MilvusException as exc:
            raise VectorStoreError(f"Milvus operation failed: {operation}") from exc

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
            f"check collection {collection_name!r}",
            has_collection,
            collection_name=collection_name,
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
                if collection_name not in self._validated_collections:
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
                await asyncio.to_thread(create_collection, **create_kwargs)
            except MilvusException as exc:
                if await self._has_collection(collection_name):
                    await self._validate_collection_schema(collection_name)
                    return collection_name
                raise VectorStoreError(
                    f"Failed to create Milvus collection {collection_name!r}"
                ) from exc

            self._validated_collections.add(collection_name)

        return collection_name

    async def _describe_collection(self, collection_name: str) -> dict[str, Any]:
        """Describe a collection using the Milvus client."""
        describe_collection = cast(
            Callable[..., dict[str, Any]], self.client.describe_collection
        )
        return await self._run_client_call(
            f"describe collection {collection_name!r}",
            describe_collection,
            collection_name=collection_name,
        )

    async def _validate_collection_schema(self, collection_name: str) -> None:
        """Validate that an existing collection matches Honcho's schema."""
        description = await self._describe_collection(collection_name)
        fields = cast(list[dict[str, Any]], description.get("fields", []))
        by_name = {str(field.get("name")): field for field in fields}

        if description.get("enable_dynamic_field") is not True:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} must enable dynamic fields"
            )
        if description.get("auto_id") is not False:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} must disable auto-generated IDs"
            )

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

        metadata_field = by_name.get(METADATA_FIELD)
        if metadata_field is None:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} is missing required JSON field"
                + f" {METADATA_FIELD!r}"
            )
        if metadata_field.get("type") != DataType.JSON:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} field {METADATA_FIELD!r}"
                + " must be JSON"
            )
        if metadata_field.get("nullable") is not True:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} field {METADATA_FIELD!r}"
                + " must be nullable"
            )

        for field_name in STANDARD_METADATA_FIELDS:
            field = by_name.get(field_name)
            if field is None:
                raise VectorStoreError(
                    f"Milvus collection {collection_name!r} is missing required field"
                    + f" {field_name!r}"
                )
            if field.get("type") != DataType.VARCHAR:
                raise VectorStoreError(
                    f"Milvus collection {collection_name!r} field {field_name!r}"
                    + " must be VARCHAR"
                )
            if field.get("nullable") is not True:
                raise VectorStoreError(
                    f"Milvus collection {collection_name!r} field {field_name!r}"
                    + " must be nullable"
                )

        actual_dim = self._extract_vector_dim(description)
        expected_dim = settings.EMBEDDING.VECTOR_DIMENSIONS
        if actual_dim != expected_dim:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} vector dim ({actual_dim})"
                + f" does not match EMBEDDING_VECTOR_DIMENSIONS ({expected_dim})"
            )

        list_indexes = cast(Callable[..., list[str]], self.client.list_indexes)
        index_names = await self._run_client_call(
            f"list indexes for collection {collection_name!r}",
            list_indexes,
            collection_name=collection_name,
        )
        vector_index: dict[str, Any] | None = None
        describe_index = cast(Callable[..., dict[str, Any]], self.client.describe_index)
        for index_name in index_names:
            index = await self._run_client_call(
                f"describe index {index_name!r} for collection {collection_name!r}",
                describe_index,
                collection_name=collection_name,
                index_name=index_name,
            )
            if index.get("field_name") == VECTOR_FIELD:
                vector_index = index
                break

        if vector_index is None:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} has no index for"
                + f" vector field {VECTOR_FIELD!r}"
            )
        if str(vector_index.get("metric_type", "")).upper() != DISTANCE_METRIC:
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} vector index must use"
                + f" {DISTANCE_METRIC} metric"
            )
        if str(vector_index.get("index_type", "")).upper() != "AUTOINDEX":
            raise VectorStoreError(
                f"Milvus collection {collection_name!r} vector index must use AUTOINDEX"
            )

        self._validated_collections.add(collection_name)

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
            try:
                return int(dim)
            except (TypeError, ValueError) as exc:
                raise VectorStoreError(
                    "Milvus collection vector dimension must be an integer"
                ) from exc
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
        await self._run_client_call(
            f"upsert {len(vectors)} vectors into namespace {namespace!r}",
            upsert,
            collection_name=collection_name,
            data=rows,
        )
        logger.debug(
            "Upserted %s vectors to Milvus namespace %s",
            len(vectors),
            namespace,
        )

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
        if collection_name not in self._validated_collections:
            await self._validate_collection_schema(collection_name)

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
        if settings.VECTOR_STORE.MILVUS_CONSISTENCY_LEVEL:
            search_kwargs["consistency_level"] = (
                settings.VECTOR_STORE.MILVUS_CONSISTENCY_LEVEL
            )
        if output_fields is not None:
            search_kwargs["output_fields"] = output_fields

        search = cast(Callable[..., list[list[dict[str, Any]]]], self.client.search)
        batches = await self._run_client_call(
            f"query namespace {namespace!r}", search, **search_kwargs
        )

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
        if "distance" not in hit:
            return 0.0

        raw_distance = float(hit["distance"])
        # Milvus Lite 3.0 reports COSINE distance while Milvus server and
        # Zilliz Cloud report COSINE similarity. Keep this workaround scoped
        # to the affected Lite release: https://github.com/milvus-io/milvus-lite/issues/343
        if _uses_lite_3_cosine_distance(settings.VECTOR_STORE.MILVUS_URI):
            return raw_distance
        return 1.0 - raw_distance

    def _output_fields(self, include_attributes: bool | list[str]) -> list[str] | None:
        """Translate Honcho projection settings to Milvus output fields."""
        if include_attributes is True:
            return [METADATA_FIELD]
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
        await self._run_client_call(
            f"delete {len(ids)} vectors from namespace {namespace!r}",
            delete,
            collection_name=collection_name,
            ids=ids,
        )
        logger.debug("Deleted %s vectors from Milvus namespace %s", len(ids), namespace)

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
        await self._run_client_call(
            f"delete namespace {namespace!r}",
            drop_collection,
            collection_name=collection_name,
        )
        self._validated_collections.discard(collection_name)
        logger.debug("Deleted Milvus namespace %s", namespace)

    async def close(self) -> None:
        """Close the Milvus client connection and release resources."""
        await self._run_client_call("close client", self.client.close)
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

        if collection_name not in self._validated_collections:
            await self._validate_collection_schema(collection_name)
        return settings.EMBEDDING.VECTOR_DIMENSIONS
