import contextlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, TypeVar, cast

from pgvector.vector import Vector
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import make_transient_to_detached

from src.cache import client
from src.cache.utils import (
    CacheKey,
    get_cache_namespace,
)
from src.config import settings

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ModelCache:
    """Wrapper for SQLAlchemy queries with Redis caching"""

    def __init__(
        self,
        resource_type: str,
        ttl: int = 300,
    ) -> None:
        self.ttl: int = ttl
        self.namespace: str = get_cache_namespace()
        self.resource_type: str = resource_type
        # LRU limits
        self.max_per_namespace: int = settings.CACHE.MAX_WORKSPACE_LIMIT

    async def get_or_fetch(
        self,
        db: AsyncSession,
        model_class: type[T],
        cache_key: str,
        query_func: Callable[[AsyncSession], Any],
    ) -> T | None:
        """
        Attempts to fetch a value from Redis first, falling back to the database if not found.

        On cache hit, the workspace LRU is updated with the current timestamp.

        Args:
            db: Database session
            model_class: SQLAlchemy model class
            cache_key: Redis key
            query_func: Async function that executes a query and returns the extracted value
        """
        cached = await client.get(cache_key) if client.is_enabled() else None
        if cached:
            try:
                raw = self._ensure_str(cached)
                data = json.loads(raw)

                await self._touch_workspace_lru(cache_key)
                return self._deserialize_model(model_class, data)
            except (json.JSONDecodeError, Exception) as e:
                # Log error but continue to DB fetch
                logger.warning(
                    "Cache deserialization error for %s: %s", model_class.__name__, e
                )

        # Fall back to database
        result = await query_func(db)

        if result and client.is_enabled():
            # Cache the result
            data = self._serialize_model(result)
            await client.set(cache_key, json.dumps(data), ttl_seconds=self.ttl)

        return result

    async def get_list_or_fetch(
        self,
        db: AsyncSession,
        model_class: type[T],
        cache_key: str,
        query_func: Callable[[AsyncSession], Any],
    ) -> list[T]:
        """
        Attempts to fetch a list of values from Redis first, falling back to the database if not found.

        On cache hit, the workspace LRU is updated with the current timestamp.

        Args:
            db: Database session
            model_class: SQLAlchemy model class
            cache_key: Redis key
            query_func: Async function that executes a query and returns the extracted values
        """
        cached = await client.get(cache_key) if client.is_enabled() else None
        if cached:
            try:
                raw = self._ensure_str(cached)
                data_list = json.loads(raw)

                await self._touch_workspace_lru(cache_key)
                return [self._deserialize_model(model_class, d) for d in data_list]
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    "Cache deserialization error for %s: %s", model_class.__name__, e
                )

        result = await query_func(db)

        if result and client.is_enabled():
            data_list = [self._serialize_model(item) for item in result]
            await client.set(cache_key, json.dumps(data_list), ttl_seconds=self.ttl)

        return result

    async def set(
        self,
        cache_key: str,
        obj: Any,
        ttl: int | None = None,
    ) -> None:
        """Explicitly set a cache entry. For workspace, also updates the workspace LRU and enforces per-namespace workspace limits.

        Args:
            cache_key: Redis key
            obj: SQLAlchemy model instance
            ttl: Time to live in seconds
        """
        if not client.is_enabled():
            return
        data = self._serialize_model(obj)

        # If caching a workspace, also set the workspace in a LRU redis sorted set
        if self.resource_type == "workspace":
            await self._workspace_lru_set_internal(
                cache_key, json.dumps(data), ttl_seconds=ttl or self.ttl
            )
        else:
            await client.set(cache_key, json.dumps(data), ttl_seconds=ttl or self.ttl)

    def _serialize_model(self, obj: Any) -> dict[str, Any]:
        """Serialize SQLAlchemy model instance to JSON-compatible dict"""

        mapper = inspect(obj.__class__)
        data: dict[str, Any] = {}
        for attr in mapper.column_attrs:
            key = attr.key
            value = getattr(obj, key)
            if value is None:
                data[key] = None
            elif hasattr(value, "isoformat"):
                data[key] = value.isoformat()
            elif isinstance(value, list | dict):
                data[key] = value
            elif isinstance(value, Vector):
                data[key] = value.to_list()
            else:
                data[key] = value
        return data

    def _deserialize_model(self, model_class: type[T], data: dict[str, Any]) -> T:
        """Reconstruct SQLAlchemy model instance from cached dict"""

        mapper = inspect(model_class)
        if mapper is None:
            return model_class(**data)

        if mapper.column_attrs:
            for attr in mapper.column_attrs:
                key = attr.key
                if key not in data:
                    continue
                value = data[key]
                column = attr.columns[0] if attr.columns else None

                if column is None or value is None:
                    continue

                if (
                    isinstance(value, str)
                    and hasattr(column.type, "python_type")
                    and column.type.python_type == datetime
                ):
                    with contextlib.suppress(ValueError):
                        data[key] = datetime.fromisoformat(data[key])
                elif (
                    isinstance(value, list)
                    and column is not None
                    and hasattr(column.type, "__class__")
                    and column.type.__class__.__name__ == "VECTOR"
                ):
                    vector_values = cast(Sequence[float | int | str | None], value)
                    floats = [float(elem) for elem in vector_values if elem is not None]
                    data[key] = Vector(floats)

        # Create the instance
        instance = model_class(**data)

        # Make it detached (not transient) so it can be merged with load=False

        make_transient_to_detached(instance)

        return instance

    async def evict_resource(
        self,
        cache_key: str,
    ) -> None:
        """
        Evicts a workspace and its associated resources from the cache.
        If the resource type is a workspace, we also clean up the workspace LRU sorted set.
        """
        await client.delete_prefix(prefix=cache_key)
        if self.resource_type == "workspace":
            await client.zrem(self.get_lru_key_workspace(), cache_key)

    def _ensure_str(self, value: Any) -> str:
        if isinstance(value, bytes | bytearray):
            return value.decode("utf-8")
        return str(value)

    # ----- Internal workspace LRU helpers -----
    def get_lru_key_workspace(self) -> str:
        """
        Get the redis sorted set key for the workspace LRU sorted set.
        """
        return f"lru:{self.namespace}:ws:recency"

    async def _touch_workspace_lru(self, cache_key: str) -> None:
        """Update workspace access timestamp on cache hit."""
        if not client.is_enabled():
            return

        workspace_lru_key = self.get_lru_key_workspace()
        now_ms = time.time() * 1000
        workspace_cache_key = CacheKey.from_cache_key(
            cache_key
        ).get_workspace_cache_key()

        await client.zadd(workspace_lru_key, {workspace_cache_key: float(now_ms)})

    async def _workspace_lru_set_internal(
        self,
        workspace_cache_key: str,
        value: bytes | str,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set workspace in cache with LRU eviction enforcement.

        Atomically writes the workspace data, updates LRU tracking, and evicts
        the least recently used workspaces if the namespace exceeds max_per_namespace.

        Uses two Redis pipelines:
        1. SET payload + ZADD timestamp + ZCARD to check size
        2. If overflow: ZPOPMIN oldest + DELETE their data (including all sub-resources)
        """
        if not client.is_enabled():
            return

        workspace_lru_key = self.get_lru_key_workspace()
        now_ms = time.time() * 1000
        pipe = client.pipeline(transaction=False)
        pipe.set(workspace_cache_key, value, ex=ttl_seconds)  # set workspace in cache
        pipe.zadd(
            workspace_lru_key, {workspace_cache_key: now_ms}
        )  # add current timestamp to lru within redis
        pipe.zcard(workspace_lru_key)  # get count of elements in the namespace LRU

        set_results: list[Any] = await pipe.execute()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        size = cast(int, set_results[2])  # number of elements in the namespace LRU
        overflow = size - self.max_per_namespace
        if overflow > 0:
            # If we are over the # of workspaces in the namespace, pop the oldest workspaces from the LRU
            pipe2 = client.pipeline(transaction=False)
            pipe2.zpopmin(workspace_lru_key, overflow)
            overflow_results: list[Any] = await pipe2.execute()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            popped = cast(list[tuple[bytes, float]], overflow_results[0])

            if popped:
                workspaces_to_evict = [key.decode("utf-8") for key, _ in popped]
                deleted_total = 0
                for workspace_cache_key in workspaces_to_evict:
                    deleted_total += await client.delete_prefix(workspace_cache_key)
