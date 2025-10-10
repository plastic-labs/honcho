from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    from redis.asyncio.client import Redis as AsyncRedis

from src.config import settings

logger = logging.getLogger(__name__)

_client: AsyncRedis | None = None
_lock = asyncio.Lock()


def _client_available() -> bool:
    return settings.CACHE.ENABLED


async def init_cache() -> None:
    """Establish the Redis connection if caching is enabled."""
    global _client
    if not _client_available():
        _client = None
        return
    if _client is None:
        async with _lock:
            if _client is None:
                _client = redis.from_url(  # pyright: ignore
                    settings.CACHE.URL,
                    encoding="utf-8",
                    decode_responses=False,
                )
                logger.info("Connected to cache at %s", settings.CACHE.URL)


async def close_cache() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def set(key: str, value: bytes | str, ttl_seconds: int | None = None) -> None:
    client = _client
    if client is None:
        return
    if isinstance(value, str):
        value = value.encode("utf-8")
    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE.DEFAULT_TTL_SECONDS
    await client.set(key, value, ex=ttl)


async def get(key: str) -> bytes | None:
    client = _client
    if client is None:
        return None
    return await client.get(key)


async def delete(key: str) -> None:
    client = _client
    if client is None:
        return
    await client.delete(key)


__all__ = [
    "init_cache",
    "set",
    "get",
    "delete",
]
