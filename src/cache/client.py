from __future__ import annotations

import asyncio
import logging

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

cache_initialized = False

_cache_lock = asyncio.Lock()


def is_cache_enabled() -> bool:
    return settings.CACHE.ENABLED


# Initialize cache at module import time (synchronously)
# This ensures ContextVars are properly initialized before any async contexts
if is_cache_enabled():
    try:
        cache.setup(  # pyright: ignore[reportUnknownMemberType]
            settings_url=settings.CACHE.URL,
            pickle_type=PicklerType.SQLALCHEMY,
        )
        logger.debug("Cache setup completed at module import")
        cache_initialized = True
    except Exception as e:
        logger.warning(
            "Failed to setup cache at module import: %s. Will retry in init_cache()",
            e,
        )
        if settings.SENTRY.ENABLED:
            sentry_sdk.capture_exception(e)
        cache.setup("mem://")  # pyright: ignore[reportUnknownMemberType]
else:
    cache.setup("mem://")  # pyright: ignore[reportUnknownMemberType]
    cache.disable()
    logger.debug("Cache disabled via settings")


async def init_cache() -> None:
    """Initialize and verify cache connection if enabled."""
    global cache_initialized
    if not is_cache_enabled():
        logger.info("Cache disabled")
        return

    async with _cache_lock:
        # If cache was not successfully initialized at module import, try again here
        if not cache_initialized:
            try:
                cache.setup(  # pyright: ignore[reportUnknownMemberType]
                    settings_url=settings.CACHE.URL,
                    pickle_type=PicklerType.SQLALCHEMY,
                )
                cache_initialized = True
            except Exception as setup_err:
                logger.warning(
                    "Cache setup failed for %s: %s", settings.CACHE.URL, setup_err
                )
                if settings.SENTRY.ENABLED:
                    sentry_sdk.capture_exception(setup_err)
                settings.CACHE.ENABLED = False
                cache.disable()
                return

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
                "Failed to connect to cache at %s: %s. Disabling cache",
                settings.CACHE.URL,
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            settings.CACHE.ENABLED = False
            cache.disable()
        except Exception as e:
            logger.warning(
                "Unexpected cache error at %s: %s. Disabling cache",
                settings.CACHE.URL,
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            settings.CACHE.ENABLED = False
            cache.disable()


async def close_cache() -> None:
    await cache.close()


__all__ = [
    "init_cache",
    "close_cache",
    "cache",
]
