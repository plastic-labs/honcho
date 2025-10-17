import contextlib
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from pgvector.vector import Vector
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient_to_detached

from src.cache import client
from src.cache.constants import (
    get_cache_namespace,
    get_peer_prefix,
    get_session_prefix,
    get_workspace_prefix,
)

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
                logger.warning(
                    "Cache deserialization error for %s: %s", model_class.__name__, e
                )

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
                logger.warning(
                    "Cache deserialization error for %s: %s", model_class.__name__, e
                )

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
    ) -> None:
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

    def construct_cache_key(
        self,
        *,
        workspace_name: str | None = None,
        session_name: str | None = None,
        peer_name: str | None = None,
    ) -> str:
        """
        Construct a cache key for workspaces, sessions, or peers.

        Args:
            workspace_name (str, optional): The workspace name
            session_name (str, optional): The session name
            peer_name (str, optional): The peer name (for peers)

        Returns:
            str: The formatted cache key.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """

        if self.resource_type == "workspace":
            if not workspace_name:
                raise ValueError("Must specify 'workspace_name' for workspace key")
            return get_workspace_prefix(workspace_name)
        elif self.resource_type == "session":
            if not (workspace_name and session_name):
                raise ValueError(
                    "Must specify both 'workspace_name' and 'session_name' for session key"
                )
            return f"{get_session_prefix(workspace_name)}:{session_name}"
        elif self.resource_type == "peer":
            if not (workspace_name and peer_name):
                raise ValueError(
                    "Must specify both 'workspace_name' and 'peer_name' for peer key"
                )
            return f"{get_peer_prefix(workspace_name)}:{peer_name}"
        else:
            raise ValueError(
                f"Unknown resource type '{self.resource_type}', must be one of: workspace, session, peer"
            )
