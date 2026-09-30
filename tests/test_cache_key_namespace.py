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
from src.config import settings
from src.crud.session import SESSION_CACHE_KEY_TEMPLATE, session_cache_key
from src.db import tenant_context


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


@pytest.mark.asyncio
async def test_cache_isolates_tenants_under_multi_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under MULTI_TENANT the tenant-scope middleware keeps entries per-tenant.

    Cache keys are workspace_name-scoped, and workspace_name is not unique across
    tenants (every tenant has a "default" workspace). Without tenant scoping a
    read-through hit would serve another tenant's row and skip RLS. Two tenants
    sharing a key must not see each other's values; each reads back its own. (This
    fails if the middleware is inactive — the shared key would collide.)
    """
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    key = session_cache_key("default", "s1")  # a name-only key that collides

    token = tenant_context.set("tenant-a")
    try:
        await cache.set(key, "value-a", expire=60)
        assert await cache.get(key) == "value-a"
    finally:
        tenant_context.reset(token)

    token = tenant_context.set("tenant-b")
    try:
        assert await cache.get(key) is None  # tenant-b can't see tenant-a's entry
        await cache.set(key, "value-b", expire=60)
        assert await cache.get(key) == "value-b"
    finally:
        tenant_context.reset(token)

    token = tenant_context.set("tenant-a")
    try:
        assert await cache.get(key) == "value-a"  # tenant-a still reads its own
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_tenant_scope_prefixes_the_key_actually_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefix lands on the key the backend stores, not just on read-back.

    Complements test_cache_isolates_tenants_under_multi_tenant (which infers the
    prefix from isolation behavior) by scanning the backend directly.
    """
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    key = session_cache_key("w1", "s1")

    token = tenant_context.set("tenant-prefix-check")
    try:
        await cache.set(key, "value", expire=60)
        written = [k async for k in cache.scan("*")]
    finally:
        tenant_context.reset(token)

    assert f"t:tenant-prefix-check:{key}" in written


@pytest.mark.asyncio
async def test_tenant_scope_raises_without_tenant_for_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed, not 't:default:...': 'default' is the bootstrap tenant's real
    id, so falling back to it would silently share that tenant's cache entries
    with every tenant-less caller."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert tenant_context.get() is None

    with pytest.raises(ValueError, match="requires a tenant"):
        await cache.get(session_cache_key("w1", "s1"))


@pytest.mark.asyncio
async def test_tenant_scope_raises_without_tenant_for_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert tenant_context.get() is None

    with pytest.raises(ValueError, match="requires a tenant"):
        await cache.set(session_cache_key("w1", "s1"), "value", expire=60)


@pytest.mark.asyncio
async def test_tenant_scope_raises_without_tenant_for_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert tenant_context.get() is None

    with pytest.raises(ValueError, match="requires a tenant"):
        await cache.delete(session_cache_key("w1", "s1"))


@pytest.mark.asyncio
async def test_tenant_scope_raises_without_tenant_for_delete_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*_many commands are scoped per key, same as the singular commands."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert tenant_context.get() is None

    with pytest.raises(ValueError, match="requires a tenant"):
        await cache.delete_many(
            session_cache_key("w1", "s1"), session_cache_key("w2", "s2")
        )


@pytest.mark.asyncio
async def test_tenant_scope_flag_off_no_prefix_and_no_raise() -> None:
    """MULTI_TENANT off: no tenant is required, and self-host keys stay unprefixed."""
    assert settings.MULTI_TENANT is False
    assert tenant_context.get() is None
    key = session_cache_key("w1", "s1")

    await cache.set(key, "value", expire=60)
    assert await cache.get(key) == "value"
    await cache.delete_many(key)

    written = [k async for k in cache.scan("*")]
    assert not any(k.startswith("t:") for k in written)
