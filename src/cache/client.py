from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import sentry_sdk
from cashews import cache
from cashews.backends.redis.client import SafeRedisCluster
from cashews.commands import PATTERN_CMDS, Command
from cashews.picklers import PicklerType
from redis import exceptions as redis_exc
from redis.asyncio import RedisCluster
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


# region ai
# Compatibility shim: cashews 7.5.0 against redis-py >= 8.0.0.
#
# REMOVE THIS once cashews ships a release that accepts the arguments redis-py
# 8.x passes to `RedisCluster.initialize()`. Check `SafeRedisCluster.initialize`
# upstream: if its signature takes *args/**kwargs (or the two parameters below),
# this block is dead weight and should go with the cashews bump.
#
# cashews 7.5.0 (2026-03-02) predates redis-py 8.0.0 (2026-05-28). redis-py
# PR #4060 added `additional_startup_nodes_info` and `last_failed_node_name` to
# `RedisCluster.initialize()`, but cashews still declares its override as
# `initialize(self)`. redis-py calls it with a keyword on the command-retry path
# (`redis/asyncio/cluster.py`, guarded by `if self._initialize`), so the call
# raises TypeError. `_initialize` is set by any ConnectionError/TimeoutError,
# which makes a single unreachable node enough to trigger it.
#
# Two reasons this matters more than a normal signature drift:
#   * TypeError is not in the tuple cashews catches (RedisError, socket.gaierror,
#     OSError, asyncio.TimeoutError), so it bypasses SafeRedisCluster's entire
#     purpose and propagates into request handling instead of degrading.
#   * `initialize()` is itself the recovery path, so it never clears
#     `_initialize` and every later command fails too. The client stays wedged
#     until the process restarts.
#
# Only reachable with CACHE.CLUSTER enabled, and only after a node failure, so
# it does not show up in standalone local stacks or in CI. Not reported upstream
# at the time of writing; a standalone reproduction lives in DEV-2647.
async def _safe_cluster_initialize(
    self: SafeRedisCluster, *args: Any, **kwargs: Any
) -> SafeRedisCluster:
    """Forward whatever redis-py passes, keeping cashews' degrade-on-error intent."""
    try:
        return await RedisCluster.initialize(self, *args, **kwargs)  # pyright: ignore[reportReturnType]
    except (
        redis_exc.RedisError,
        socket.gaierror,
        OSError,
        TimeoutError,
    ):
        logger.error("redis: can not initialize cache", exc_info=True)
        return self


# cashews evaluates `__aenter__ = initialize` at class-definition time, so the
# alias still references the original function. Patching only `initialize` would
# leave the async-context-manager path broken.
SafeRedisCluster.initialize = _safe_cluster_initialize
SafeRedisCluster.__aenter__ = _safe_cluster_initialize
# endregion


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


# On Redis Cluster a key's slot is derived from the substring inside the first
# {...}, when one is present. Tagging the namespace puts every key an instance
# writes on a single slot, and therefore a single shard, so its client holds
# connections to one node rather than to all of them. Namespaces still hash
# independently of one another, so keys stay spread across the cluster.
#
# Two spellings, because the two ways a key gets built treat the string
# differently: cashews runs `prefix=` through format substitution, so braces
# have to be doubled to survive as literals, while direct construction does no
# substitution and needs them single. Both render to the same bytes, which
# tests/cache/test_cache_namespace_hash_tag.py asserts -- a mismatch would send
# writes and deletes to different keys with nothing raised.
def cache_key_namespace() -> str:
    """Tagged namespace for keys built by string concatenation."""
    return "{" + get_cache_namespace() + "}"


def cache_prefix_namespace() -> str:
    """Tagged namespace for cashews `prefix=`, which format-substitutes."""
    return "{{" + get_cache_namespace() + "}}"


def _tenant_scope_middleware() -> Any:
    """Prefix every cache key with the current tenant when MULTI_TENANT is on."""

    # region ai
    # honcho's cache keys are workspace_name-scoped, and workspace_name is not unique
    # across tenants (every tenant has a "default" workspace), so without this a
    # cross-tenant cache hit would return another tenant's row and bypass row-level
    # security — the cache is read before the DB. Prefixing every key with the
    # request's tenant keeps entries (and the per-key locks) isolated across
    # get/set/delete. No-op when MULTI_TENANT is off, so self-host keys are
    # byte-for-byte unchanged. Modeled on cashews' own add_prefix helper.
    # endregion

    async def _middleware(
        call: Any, cmd: Command, _backend: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if not settings.MULTI_TENANT:
            return await call(*args, **kwargs)
        # ai: deferred import keeps cache.client free of a load-time dependency on db.
        from src.db import tenant_context

        tenant = tenant_context.get()

        def _scope(key: str) -> str:
            # region ai
            # Fail closed instead of falling back to a shared 'default' bucket:
            # 'default' is the real id of the bootstrap tenant (see the tenants
            # seed migration), so a tenant-less caller reading/writing
            # t:default:* would silently share that tenant's cache -- and since
            # the cache is read before the DB, a cross-tenant hit here bypasses
            # RLS entirely, the same class of bug tracked_db() and
            # construct_work_unit_key() already fail closed on. Every reachable
            # caller is tenant-bound before it can reach a cache command: those
            # two raise first for every path that reaches this middleware
            # (tracked_db() for every tracked_db-scoped call, and
            # construct_work_unit_key() for the one service_db()-scoped cache
            # write, the scope-task enqueue's cache invalidation). If this
            # raises, some new caller reached the cache without going through
            # either, and belongs on tracked_db(tenant_id=...) instead.
            # endregion
            if not tenant:
                raise ValueError(
                    f"cache {cmd.value} on key {key!r} requires a tenant when "
                    + "MULTI_TENANT is on, but tenant_context is unset -- bind a "
                    + "tenant (tracked_db(tenant_id=...), or an ambient "
                    + "tenant_context set by the caller) before touching the "
                    + "cache; cross-tenant cache access is not supported"
                )
            return f"t:{tenant}:" + key

        if cmd in (Command.GET_MANY, Command.DELETE_MANY):
            return await call(*[_scope(key) for key in args])
        if cmd == Command.SET_MANY:
            kwargs["pairs"] = {_scope(k): v for k, v in kwargs["pairs"].items()}
            return await call(**kwargs)
        as_key = "pattern" if cmd in PATTERN_CMDS else "key"
        key = kwargs.get(as_key)
        if key:
            kwargs[as_key] = _scope(key)
            return await call(**kwargs)
        if args:
            return await call(_scope(args[0]), *args[1:], **kwargs)
        return await call(*args, **kwargs)

    return _middleware


# ai: registered once at import; applies to every backend cache.setup() installs.
cache.add_middleware(_tenant_scope_middleware())


async def _release_default_node_connection() -> None:
    """Drop the idle connection the startup PING leaves on the cluster's default node.

    PING carries no key, so redis-py routes it to ``nodes_manager.default_node``
    rather than to a shard. That node is the same one for every client in a
    deployment: ``NodesManager.initialize`` fills ``nodes_cache`` in the order
    CLUSTER SLOTS returns slot ranges -- ascending since Redis 6.2 -- and then
    takes ``get_nodes_by_server_type(PRIMARY)[0]``, so everyone picks whichever
    primary owns slot 0.

    Nothing closes that connection afterwards. It is returned to the pool on
    release, redis-py's cluster node pool does no idle reaping, and every later
    cache call is keyed and hash-tagged to this instance's namespace, so it goes
    to a data shard instead. The result is one permanently idle socket per
    process, all of them on one node, while the keyed traffic spreads evenly.

    At fleet scale that dominates the connection count: the slot-0 primary held
    roughly 13x the connections of its peers, ~92% of them idle since their
    PING, even though the three primaries' keyed traffic was within 13% of each
    other. Enough to reach ``maxclients`` on that one node while the rest of the
    cluster sat near a quarter of it, which fails new clients everywhere --
    ``RedisCluster.initialize`` cannot complete if the default node refuses the
    connection, so the client never reaches the shard holding its own keys.

    Safe because it only disconnects free connections and leaves them in the
    pool: redis-py re-establishes lazily if a keyless command is ever issued
    again. A topology refresh re-opens one, since the refreshed client asks the
    default node for the command table, but that is per-refresh rather than for
    the life of the process.
    """
    if not settings.CACHE.CLUSTER:
        return
    try:
        # No public accessor reaches the node objects, so this reads through
        # cashews' backend to the redis-py client it wraps. Guarded below
        # because a rename in either library must not break startup: the
        # connection this releases is an optimisation, not a correctness need.
        for backend in cache._backends.values():  # pyright: ignore[reportPrivateUsage]
            default_node = getattr(
                getattr(getattr(backend, "_client", None), "nodes_manager", None),
                "default_node",
                None,
            )
            if default_node is not None:
                await default_node.disconnect_free_connections()
    except Exception:
        logger.debug(
            "Could not release the cache default-node connection", exc_info=True
        )


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
            logger.error(
                "Cache setup failed for %s: %s. Falling back to a process-local in-memory cache; invalidations will not reach other processes",
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
        except (redis_exc.TimeoutError, redis_exc.ConnectionError, TimeoutError) as e:
            logger.error(
                "Failed to connect to cache at %s: %s. Falling back to a process-local in-memory cache; invalidations will not reach other processes",
                _redact_cache_url(settings.CACHE.URL),
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            # Fallback to in-memory cache
            await cache.close()
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)
        except Exception as e:
            logger.error(
                "Unexpected cache error at %s: %s. Falling back to a process-local in-memory cache; invalidations will not reach other processes",
                _redact_cache_url(settings.CACHE.URL),
                e,
            )
            if settings.SENTRY.ENABLED:
                sentry_sdk.capture_exception(e)
            # Fallback to in-memory cache
            await cache.close()
            cache.setup("mem://", pickle_type=PicklerType.SQLALCHEMY)

        # Outside the try above deliberately: its handlers fall back to the
        # in-memory cache, and losing Redis caching process-wide is far worse
        # than leaving one idle connection behind. A no-op on the fallback
        # path, where there is no cluster client to read.
        await _release_default_node_connection()


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
    "cache_key_namespace",
    "cache_prefix_namespace",
    "safe_cache_delete",
    "safe_cache_set",
]
