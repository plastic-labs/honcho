"""The namespace hash tag must survive both ways a cache key gets built.

cashews format-substitutes `prefix=`, direct construction does not, so the two
need different spellings of the same tag. If they ever diverge, a write and its
invalidation land on different keys and nothing raises -- the cache just serves
stale rows. These tests are what fails instead.
"""

import pytest
from redis.crc import key_slot

from src.cache.client import (
    cache,
    cache_key_namespace,
    cache_prefix_namespace,
    get_cache_namespace,
)
from src.crud.session import SESSION_CACHE_KEY_TEMPLATE, session_cache_key


def test_both_spellings_render_the_same_tag():
    ns = get_cache_namespace()
    assert cache_key_namespace() == "{" + ns + "}"
    # Doubled braces collapse to single ones when cashews formats the prefix.
    assert cache_prefix_namespace().format() == cache_key_namespace()


@pytest.mark.asyncio
async def test_decorator_key_matches_helper_key():
    """The key the decorator writes is the key the helper computes."""

    @cache(
        key=SESSION_CACHE_KEY_TEMPLATE,
        prefix=cache_prefix_namespace(),
        ttl="60s",
    )
    async def get_session(workspace_name: str, session_name: str) -> str:
        # The names matter: cashews fills the key template from them.
        return f"{workspace_name}/{session_name}"

    await get_session(workspace_name="w1", session_name="s1")

    written = [k async for k in cache.scan("*")]
    assert session_cache_key("w1", "s1") in written


def test_one_namespace_hashes_to_one_slot():
    """Every key an instance writes shares a Redis Cluster slot."""
    keys = [
        session_cache_key("w1", "s1"),
        session_cache_key("w2", "s2"),
        f"{cache_key_namespace()}:lock:v2:anything",
    ]
    assert len({key_slot(k.encode()) for k in keys}) == 1


def test_namespaces_hash_independently():
    """Tagging must not collapse the whole fleet onto one shard."""
    slots = {
        key_slot(("{" + n + "}:v2:workspace:w").encode()) for n in ("a1", "b2", "c3")
    }
    assert len(slots) > 1
