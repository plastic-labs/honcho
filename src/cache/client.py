from __future__ import annotations

import asyncio
import logging
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import sentry_sdk
from cashews import cache
from cashews.picklers import PicklerType
from redis import exceptions as redis_exc
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from src.config import settings

logger = logging.getLogger(__name__)

_cache_lock = asyncio.Lock()


# Query parameters that carry secrets when configured via URL:
# redis-py accepts ``?password=`` (all querystring options become client
# kwargs) and cashews accepts ``?secret=`` (HMAC key for value signing).
_SENSITIVE_QUERY_PARAMS = frozenset({"password", "secret"})


def _mask_sensitive_query(query: str) -> str:
    """Mask values of secret-bearing query parameters.

    Operates on the raw query string (no decode/re-encode round trip)
    so non-secret parameters are preserved byte-for-byte.

    Args:
        query: The raw query string from a parsed URL.

    Returns:
        The query string with sensitive values replaced by ``***``, or
        the original string if no sensitive parameter is present.
    """
    if not query:
        return query
    parts: list[str] = []
    changed = False
    for part in query.split("&"):
        name, sep, _value = part.partition("=")
        if sep and name.lower() in _SENSITIVE_QUERY_PARAMS:
            parts.append(f"{name}=***")
            changed = True
        else:
            parts.append(part)
    return "&".join(parts) if changed else query


def _redact_cache_url(url: str) -> str:
    """Mask credentials in a Redis connection URL before logging.

    Given ``redis://:password@host:port/db`` returns
    ``redis://:***@host:port/db``; secret-bearing query parameters
    (``?password=``, ``?secret=``) are masked as well.  A URL carrying
    no credentials is returned unchanged.  This function never raises
    and never returns a credential: an invalid port is omitted from
    the output, and a URL that cannot be parsed at all is replaced by
    a generic placeholder rather than echoed back, so that logging
    inside ``except`` blocks can neither crash startup nor leak the
    secrets this helper exists to hide.

    Args:
        url: The Redis connection URL to redact.

    Returns:
        The URL with its credentials masked, the original URL if it
        carries none, or ``"<redacted-unparseable-url>"`` if parsing
        fails entirely.
    """
    try:
        parsed = urlparse(url)
        query = _mask_sensitive_query(parsed.query)
        # .password only splits netloc and never raises, unlike .port
        if parsed.password is None and query == parsed.query:
            # A string with an "@" but no parsed authority (e.g. a URL
            # missing its scheme, ":pass@host:6379/0") may still carry
            # userinfo that urlparse could not see — never echo it.
            if "@" in url and not parsed.netloc:
                return "<redacted-unparseable-url>"
            return url
        netloc = parsed.netloc
        if parsed.password is not None:
            userinfo = parsed.username or ""
            hostname = parsed.hostname or ""
            # Preserve IPv6 brackets (urlparse strips them from .hostname)
            if hostname and ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            netloc = f"{userinfo}:***@{hostname}"
            try:
                port = parsed.port
            except ValueError:
                # Invalid or out-of-range port: omit it rather than let
                # the outer fallback echo the raw URL (and its password)
                # back.
                port = None
            if port is not None:
                netloc += f":{port}"
        parsed = parsed._replace(netloc=netloc, query=query)
        return urlunparse(parsed)
    except (ValueError, TypeError):
        # Unparseable URL: never return the raw input — it may contain
        # the very password this helper exists to hide.
        return "<redacted-unparseable-url>"


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

        # Setup cache with Redis backend. CACHE_CLUSTER selects the
        # cluster-aware client, which follows the MOVED redirects a Redis
        # Cluster returns for keys hashed to another shard; the standalone
        # client treats those as command errors.
        try:
            cache.setup(
                settings.CACHE.URL,
                pickle_type=PicklerType.SQLALCHEMY,
                cluster=settings.CACHE.CLUSTER,
            )

        except Exception as setup_err:
            logger.warning(
                "Cache setup failed for %s: %s. Falling back to in-memory cache",
                _redact_cache_url(settings.CACHE.URL),
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
                        logger.info(
                            "Connected to cache at %s",
                            _redact_cache_url(settings.CACHE.URL),
                        )
        except (
            redis_exc.TimeoutError,
            redis_exc.ConnectionError,
            asyncio.TimeoutError,
            TimeoutError,
        ) as e:
            logger.warning(
                "Failed to connect to cache at %s: %s. Falling back to in-memory cache",
                _redact_cache_url(settings.CACHE.URL),
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
                _redact_cache_url(settings.CACHE.URL),
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            # Fallback to in-memory cache
            await cache.close()
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)


_TRANSIENT_CACHE_ERRORS = (
    redis_exc.TimeoutError,
    redis_exc.ConnectionError,
    asyncio.TimeoutError,
    TimeoutError,
)


async def safe_cache_set(key: str, value: Any, expire: int | float) -> None:
    """Best-effort cache set with retries on transient errors. Failures are logged but never propagate."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=0.1, max=0.5),
            retry=retry_if_exception_type(_TRANSIENT_CACHE_ERRORS),
            reraise=True,
        ):
            with attempt:
                await cache.set(key, value, expire=expire)
    except Exception:
        logger.warning("Cache set failed for key %s", key, exc_info=True)


async def safe_cache_delete(key: str) -> None:
    """Best-effort cache delete with retries on transient errors. Failures are logged but never propagate."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=0.1, max=0.5),
            retry=retry_if_exception_type(_TRANSIENT_CACHE_ERRORS),
            reraise=True,
        ):
            with attempt:
                await cache.delete(key)
    except Exception:
        logger.warning("Cache delete failed for key %s", key, exc_info=True)


async def close_cache() -> None:
    await cache.close()


__all__ = [
    "init_cache",
    "close_cache",
    "cache",
    "safe_cache_delete",
    "safe_cache_set",
]
