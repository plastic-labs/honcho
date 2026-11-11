"""Guards the cashews/redis-py cluster compatibility shim in src.cache.client.

cashews 7.5.0 declares `SafeRedisCluster.initialize(self)`, but redis-py >= 8.0.0
calls it with keyword arguments on its command-retry path. Without the shim the
call raises TypeError, which cashews does not catch, so it escapes the safe
client and wedges it until the process restarts.

If cashews is upgraded to a release that fixes this upstream, these tests still
pass and the shim can be deleted -- see `test_shim_is_still_required`, which is
the signal for when that is true.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import cashews.backends.redis.client as cashews_client
import pytest
from cashews.backends.memory import Memory
from cashews.backends.redis.client import SafeRedisCluster
from cashews.picklers import PicklerType
from redis.asyncio import RedisCluster

# Importing it also installs the shim.
from src.cache import client as cache_client
from src.config import settings

# The parameters redis-py >= 8.0.0 passes to RedisCluster.initialize().
UPSTREAM_INITIALIZE_KWARGS = ("additional_startup_nodes_info", "last_failed_node_name")


@pytest.mark.parametrize("kwarg", UPSTREAM_INITIALIZE_KWARGS)
def test_initialize_accepts_upstream_kwargs(kwarg: str):
    """The patched override must bind whatever redis-py hands it."""
    signature = inspect.signature(SafeRedisCluster.initialize)
    # Raises TypeError on an unpatched cashews 7.5.0.
    signature.bind(None, **{kwarg: "some-node:6379"})


def test_aenter_is_patched_alongside_initialize():
    """cashews aliases `__aenter__ = initialize` at class-definition time.

    Patching only `initialize` leaves the async-context-manager path pointing at
    the original, broken function, so both attributes have to be replaced.
    """
    assert (
        SafeRedisCluster.__dict__["__aenter__"]
        is SafeRedisCluster.__dict__["initialize"]
    )
    signature = inspect.signature(SafeRedisCluster.__aenter__)
    signature.bind(None, last_failed_node_name="some-node:6379")


def test_shim_is_still_required():
    """Fails once cashews widens its own override, as the cue to drop the shim.

    Reads the installed cashews source rather than the class attribute, because
    importing src.cache.client has already replaced the attribute.
    """
    source = Path(cashews_client.__file__).read_text(encoding="utf-8")
    assert "async def initialize(self):" in source, (
        "cashews no longer declares the narrow `initialize(self)` override. The "
        "SafeRedisCluster patch in src/cache/client.py is likely obsolete -- "
        "verify against the installed cashews and delete it."
    )


@pytest.mark.asyncio
async def test_startup_with_an_unreachable_cluster_falls_back_to_memory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """init_cache() retries the ping, then switches to the in-memory cache without raising."""
    original_url = settings.CACHE.URL
    attempts = 0
    patched_initialize = SafeRedisCluster.initialize

    async def counting_initialize(self: SafeRedisCluster, *args: Any, **kwargs: Any):
        nonlocal attempts
        attempts += 1
        return await patched_initialize(self, *args, **kwargs)

    monkeypatch.setattr(SafeRedisCluster, "initialize", counting_initialize)
    monkeypatch.setattr(settings.CACHE, "ENABLED", True)
    monkeypatch.setattr(settings.CACHE, "CLUSTER", True)
    monkeypatch.setattr(settings.CACHE, "URL", "redis://127.0.0.1:1?suppress=true")
    monkeypatch.setattr(settings.CACHE, "CONNECT_RETRIES", 0)
    monkeypatch.setattr(settings.SENTRY, "ENABLED", False)
    try:
        with caplog.at_level(logging.ERROR, logger="src.cache.client"):
            await cache_client.init_cache()

        assert isinstance(cache_client.cache._get_backend(""), Memory)  # pyright: ignore[reportPrivateUsage]
        assert attempts >= 2, "the startup ping is retried"
        assert any(
            "Failed to connect to cache" in r.getMessage() for r in caplog.records
        ), "the connection-error branch, not the unexpected-error one"
    finally:
        monkeypatch.undo()
        await cache_client.cache.close()
        cache_client.cache.setup(original_url, pickle_type=PicklerType.SQLALCHEMY)


def test_upstream_still_passes_the_kwargs_the_shim_forwards():
    """Sanity-check the other half: redis-py still has the wider signature."""
    upstream = inspect.signature(RedisCluster.initialize)
    assert all(name in upstream.parameters for name in UPSTREAM_INITIALIZE_KWARGS)
