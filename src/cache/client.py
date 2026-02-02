from __future__ import annotations

import asyncio
import logging
from typing import cast

import sentry_sdk
from cashews import cache
from cashews.picklers import PicklerType
from redis import exceptions as redis_exc
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from src.config import settings

logger = logging.getLogger(__name__)


_cache_lock = asyncio.Lock()


def is_cache_enabled() -> bool:
    return settings.CACHE.ENABLED


def get_cache_namespace() -> str:
    # CACHE.NAMESPACE is guaranteed to be non-None by AppSettings.propagate_namespace validator
    return cast(str, settings.CACHE.NAMESPACE)


async def init_cache() -> None:
    """Initialize and verify cache connection if enabled."""
    async with _cache_lock:
        # Close existing backends to force recreation with new ContextVars
        await cache.close()

        if not is_cache_enabled():
            # Use in-memory cache when caching is disabled
            logger.info("Cache disabled, using in-memory cache")
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)
            return

        # Setup cache with Redis backend
        try:
            cache.setup(
                settings.CACHE.URL,
                pickle_type=PicklerType.SQLALCHEMY,
            )

        except Exception as setup_err:
            logger.warning(
                "Cache setup failed for %s: %s. Falling back to in-memory cache",
                settings.CACHE.URL,
                setup_err,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(setup_err)
            # Fallback to in-memory cache
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)
            return

        cache.enable()
        # Retry Redis ping with exponential backoff
        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential_jitter(initial=0.2, max=2.0),
                stop=stop_after_delay(5),  # give it a bit more headroom
                retry=retry_if_exception_type(
                    (
                        redis_exc.TimeoutError,
                        redis_exc.ConnectionError,
                        asyncio.TimeoutError,
                        TimeoutError,
                    )
                ),
                reraise=True,
            ):
                with attempt:
                    async with asyncio.timeout(2):
                        await cache.ping()
                        logger.info("Connected to cache at %s", settings.CACHE.URL)
        except (
            redis_exc.TimeoutError,
            redis_exc.ConnectionError,
            asyncio.TimeoutError,
            TimeoutError,
        ) as e:
            logger.warning(
                "Failed to connect to cache at %s: %s. Falling back to in-memory cache",
                settings.CACHE.URL,
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            # Fallback to in-memory cache
            await cache.close()
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)
        except Exception as e:
            logger.warning(
                "Unexpected cache error at %s: %s. Falling back to in-memory cache",
                settings.CACHE.URL,
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            # Fallback to in-memory cache
            await cache.close()
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)


async def close_cache() -> None:
    await cache.close()


__all__ = [
    "init_cache",
    "close_cache",
    "cache",
]
