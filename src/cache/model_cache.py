import contextlib
import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from pgvector.vector import Vector
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src.cache import client

T = TypeVar("T")


class ModelCache:
    """Wrapper for SQLAlchemy queries with Redis caching"""

    def __init__(self, ttl: int = 300):
        self.ttl: int = ttl

    async def get_or_fetch(
        self,
        db: AsyncSession,
        model_class: type[T],
        cache_key: str,
        query_func: Callable[[AsyncSession], Any],
    ) -> T | None:
        """
        Try Redis first, fall back to DB query

        Args:
            db: Database session
            model_class: SQLAlchemy model class
            cache_key: Redis key
            query_func: Async function that returns SQLAlchemy query or executes it
        """
        cached = await client.get(cache_key) if client.is_enabled() else None
        if cached:
            try:
                raw = self._ensure_str(cached)
                data = json.loads(raw)

                return self._deserialize_model(model_class, data)
            except (json.JSONDecodeError, Exception) as e:
                # Log error but continue to DB fetch
                print(f"Cache deserialization error: {e}")

        # Fall back to database
        result = await query_func(db)

        # Handle both query objects and executed results
        if hasattr(result, "scalar_one_or_none"):
            obj = await result.scalar_one_or_none()
        elif hasattr(result, "one_or_none"):
            obj = await result.one_or_none()
        else:
            obj = result

        if obj and client.is_enabled():
            # Cache the result
            data = self._serialize_model(obj)
            await client.set(cache_key, json.dumps(data), ttl_seconds=self.ttl)

        return obj

    async def get_list_or_fetch(
        self,
        db: AsyncSession,
        model_class: type[T],
        cache_key: str,
        query_func: Callable[[AsyncSession], Any],
    ) -> list[T]:
        """Same as get_or_fetch but for list results"""
        cached = await client.get(cache_key) if client.is_enabled() else None
        if cached:
            try:
                raw = self._ensure_str(cached)
                data_list = json.loads(raw)
                return [self._deserialize_model(model_class, d) for d in data_list]
            except (json.JSONDecodeError, Exception) as e:
                print(f"Cache deserialization error: {e}")

        result = await query_func(db)

        if hasattr(result, "scalars"):
            objs = (await result.scalars()).all()
        elif hasattr(result, "all"):
            objs = await result.all()
        else:
            objs = result

        if objs and client.is_enabled():
            data_list = [self._serialize_model(obj) for obj in objs]

            await client.set(cache_key, json.dumps(data_list), ttl_seconds=self.ttl)

        return objs

    async def set(
        self,
        cache_key: str,
        obj: Any,
        ttl: int | None = None,
    ):
        """Explicitly set a cache entry"""
        if not client.is_enabled():
            return

        data = self._serialize_model(obj)
        await client.set(cache_key, json.dumps(data), ttl_seconds=ttl or self.ttl)

    async def invalidate(self, *cache_keys: str) -> None:
        """Invalidate one or more cache keys"""
        if cache_keys and client.is_enabled():
            await client.delete(*cache_keys)

    def _serialize_model(self, obj: Any) -> dict[str, Any]:
        """Default serialization using SQLAlchemy instance state"""
        from sqlalchemy.inspection import inspect

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
        """Reconstruct model instance from dict"""
        from datetime import datetime

        from sqlalchemy.inspection import inspect

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

    def _ensure_str(self, value: Any) -> str:
        if isinstance(value, bytes | bytearray):
            return value.decode("utf-8")
        return str(value)
