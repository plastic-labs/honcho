"""Qdrant vector store implementation."""

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from src.config import settings
from src.exceptions import VectorStoreError

from . import VectorQueryResult, VectorRecord, VectorStore

logger = logging.getLogger(__name__)


# Qdrant only allows UUIDs and +ve integers as point IDs.
# Ref: https://qdrant.tech/documentation/manage-data/points/#point-ids
# So we convert arbitrary strings to deterministic UUIDs.
def _point_id(string_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore. Each namespace maps to a collection."""

    _client: AsyncQdrantClient
    _vector_size: int

    def __init__(self) -> None:
        super().__init__()
        self._client = AsyncQdrantClient(
            url=settings.VECTOR_STORE.QDRANT_URL,
            api_key=settings.VECTOR_STORE.QDRANT_API_KEY,
            prefer_grpc=settings.VECTOR_STORE.QDRANT_PREFER_GRPC,
            grpc_port=settings.VECTOR_STORE.QDRANT_GRPC_PORT,
            https=settings.VECTOR_STORE.QDRANT_HTTPS,
            prefix=settings.VECTOR_STORE.QDRANT_PREFIX,
            timeout=settings.VECTOR_STORE.QDRANT_TIMEOUT,
        )
        self._vector_size = settings.VECTOR_STORE.DIMENSIONS

    async def _ensure_collection(self, name: str) -> None:
        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter | None:
        conditions: list[models.Condition] = []
        for k, v in filters.items():
            if isinstance(v, dict) and "in" in v:
                conditions.append(
                    models.FieldCondition(key=k, match=models.MatchAny(any=v["in"]))  # pyright: ignore[reportUnknownArgumentType]
                )
            else:
                conditions.append(
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))  # pyright: ignore[reportArgumentType]
                )
        return models.Filter(must=conditions) if conditions else None

    async def upsert_many(self, namespace: str, vectors: list[VectorRecord]) -> None:
        if not vectors:
            return
        await self._ensure_collection(namespace)
        points = [
            models.PointStruct(
                id=_point_id(v.id),
                vector=v.embedding,
                payload={"_id": v.id, **v.metadata},
            )
            for v in vectors
        ]
        try:
            await self._client.upsert(collection_name=namespace, points=points)
        except Exception as e:
            logger.exception(
                f"Failed to upsert {len(vectors)} vectors to namespace {namespace}"
            )
            raise VectorStoreError(
                f"Qdrant upsert failed for namespace {namespace}"
            ) from e

    async def query(
        self,
        namespace: str,
        embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[VectorQueryResult]:
        if not await self._client.collection_exists(namespace):
            return []

        response = await self._client.query_points(
            collection_name=namespace,
            query=embedding,
            limit=top_k,
            query_filter=self._build_filter(filters) if filters else None,
            with_payload=True,
        )

        results: list[VectorQueryResult] = []
        for hit in response.points:
            dist = 1.0 - float(hit.score)
            if max_distance is not None and dist > max_distance:
                continue
            payload = dict(hit.payload or {})
            results.append(
                VectorQueryResult(
                    id=payload.pop("_id", str(hit.id)),
                    score=dist,
                    metadata=payload,
                )
            )
        return results

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        if not ids:
            return
        if not await self._client.collection_exists(namespace):
            return
        await self._client.delete(
            collection_name=namespace,
            points_selector=[_point_id(i) for i in ids],
        )

    async def delete_namespace(self, namespace: str) -> None:
        if await self._client.collection_exists(namespace):
            await self._client.delete_collection(namespace)

    async def close(self) -> None:
        await self._client.close()
