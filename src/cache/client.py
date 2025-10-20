from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import redis.asyncio as redis

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis
    from redis.asyncio.client import Pipeline

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

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
                _client = redis.Redis.from_url(
                    settings.CACHE.URL,
                    encoding="utf-8",
                    decode_responses=False,
                )

                # Retry initial ping with exponential backoff
                async for attempt in AsyncRetrying(
                    wait=wait_exponential_jitter(initial=0.2, max=2.0),
                    stop=stop_after_delay(5),
                    retry=retry_if_exception_type(
                        (redis.ConnectionError, redis.TimeoutError)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        async with asyncio.timeout(2):
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
    if ttl_seconds is None:
        await _client.set(key, value)
    else:
        await _client.set(key, value, ex=ttl_seconds)


async def get(key: str) -> bytes | None:
    global _client
    if _client is None:
        return None
    return await _client.get(key)


def pipeline(transaction: bool = True) -> Pipeline[bytes]:
    """
    Create a Redis pipeline to send multiple commands to the Redis server in one transmission.
    """
    global _client
    if _client is None:
        raise ValueError("Redis client is not initialized")
    return _client.pipeline(transaction=transaction)


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
        print(f"Deleting key {key}")
        if len(keys_to_delete) >= batch_size:
            deleted_total += await _client.delete(*keys_to_delete)
            keys_to_delete.clear()

    if keys_to_delete:
        deleted_total += await _client.delete(*keys_to_delete)

    return deleted_total


async def zadd(
    key: str,
    members: dict[str | bytes, float] | Mapping[str | bytes, float],
) -> int:
    """
    Add one or more members to a sorted set, or update its score if it already exists.

    Members may be str or bytes; str are encoded as UTF-8 to match decode_responses=False.
    """
    global _client
    if _client is None:
        return 0

    encoded_members: dict[bytes, float] = {}
    for m, s in dict(members).items():
        if isinstance(m, str):
            encoded_members[m.encode("utf-8")] = s
        else:
            encoded_members[m] = s

    return await _client.zadd(
        key,
        mapping=encoded_members,  # pyright: ignore[reportArgumentType]
    )


async def zrem(key: str, *members: str | bytes) -> int:
    """Remove one or more members from a sorted set. Returns count removed."""
    global _client
    if _client is None or not members:
        return 0

    encoded: list[bytes] = [
        m.encode("utf-8") if isinstance(m, str) else m for m in members
    ]
    return await _client.zrem(key, *encoded)


__all__ = [
    "init_cache",
    "close_cache",
    "set",
    "get",
    "delete",
    "delete_prefix",
    "zadd",
    "zrem",
    "is_enabled",
]
