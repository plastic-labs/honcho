from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis

from src.config import settings

logger = logging.getLogger(__name__)

_client: AsyncRedis[bytes] | None = None
_lock = asyncio.Lock()


def is_enabled() -> bool:
    return settings.CACHE.ENABLED


async def init_cache() -> None:
    """Establish the Redis connection if caching is enabled."""
    global _client
    if not is_enabled():
        _client = None
        return

    if _client is not None:
        return

    async with _lock:
        if _client is None:
            try:
                _client = redis.Redis[bytes].from_url(
                    settings.CACHE.URL,
                    encoding="utf-8",
                    decode_responses=False,
                )
                async with asyncio.timeout(5):
                    await _client.ping()
                    logger.info("Connected to cache at %s", settings.CACHE.URL)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout connecting to cache at %s. Disabling cache",
                    settings.CACHE.URL,
                )
                disable_cache()
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(
                    "Failed to connect to cache at %s: %s. Disabling cache",
                    settings.CACHE.URL,
                    e,
                )
                disable_cache()
            except Exception as e:
                logger.warning(
                    "Unexpected error connecting to cache at %s: %s. Disabling cache",
                    settings.CACHE.URL,
                    e,
                )
                disable_cache()


def disable_cache() -> None:
    global _client
    settings.CACHE.ENABLED = False
    _client = None


async def close_cache() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def set(key: str, value: bytes | str, ttl_seconds: int | None = None) -> None:
    global _client
    if _client is None:
        return
    if isinstance(value, str):
        value = value.encode("utf-8")
    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE.DEFAULT_TTL_SECONDS
    await _client.set(key, value, ex=ttl)


async def get(key: str) -> bytes | None:
    global _client
    if _client is None:
        return None
    return await _client.get(key)


async def delete(*keys: str) -> int:
    global _client
    if _client is None or not keys:
        return 0
    return await _client.delete(*keys)


async def delete_prefix(prefix: str, *, batch_size: int = 500) -> int:
    """Delete all keys matching a prefix using SCAN and DEL batches."""
    global _client
    if _client is None:
        return 0

    pattern = f"{prefix}*"
    keys_to_delete: list[bytes] = []
    deleted_total = 0

    async for key in _client.scan_iter(match=pattern, count=batch_size):
        keys_to_delete.append(key)
        if len(keys_to_delete) >= batch_size:
            deleted_total += await _client.delete(*keys_to_delete)
            keys_to_delete.clear()

    if keys_to_delete:
        deleted_total += await _client.delete(*keys_to_delete)

    return deleted_total


__all__ = [
    "init_cache",
    "close_cache",
    "set",
    "get",
    "delete",
    "delete_prefix",
    "is_enabled",
]
