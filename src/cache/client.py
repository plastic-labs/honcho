from __future__ import annotations

import logging

from cashews import cache
from cashews.picklers import PicklerType

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize cache at module import time (synchronously)
# This ensures ContextVars are properly initialized before any async contexts
if settings.CACHE.ENABLED:
    try:
        cache.setup(  # pyright: ignore[reportUnknownMemberType]
            settings_url=settings.CACHE.URL,
            pickle_type=PicklerType.SQLALCHEMY,  # Use sqlalchemy for SQLAlchemy object serialization
        )
        logger.debug("Cache setup completed at module import")
    except Exception as e:
        logger.warning(
            "Failed to setup cache at module import: %s. Will retry during init_cache()",
            e,
        )
        cache.setup("mem://")  # pyright: ignore[reportUnknownMemberType]
        cache.disable()
else:
    cache.setup("mem://")  # pyright: ignore[reportUnknownMemberType]
    cache.disable()
    logger.debug("Cache disabled via settings")


def is_enabled() -> bool:
    return settings.CACHE.ENABLED


async def init_cache() -> None:
    """Test cache connection if caching is enabled."""
    if not is_enabled():
        logger.info("Cache disabled")
        return

    try:
        # Test connection
        await cache.ping()
        logger.info("Connected to cache at %s", settings.CACHE.URL)
    except Exception as e:
        logger.warning(
            "Failed to connect to cache at %s: %s. Disabling cache",
            settings.CACHE.URL,
            e,
        )
        settings.CACHE.ENABLED = False
        cache.disable()


async def close_cache() -> None:
    await cache.close()


__all__ = [
    "init_cache",
    "close_cache",
    "is_enabled",
    "cache",
]
