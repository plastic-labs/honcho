"""Tests for the tenant-scoped vector namespace prefix.

Covers `src/vector_store/tenant_namespace.py` -- `resolve_namespace_prefix` and
`prefix_for_tenant`, the process-cached, per-tenant replacement for the single
process-wide `namespace_prefix` the vector store used to read once at construction.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.db import tenant_context
from src.exceptions import VectorNamespaceUnresolved
from src.vector_store import _hash_namespace_components
from src.vector_store import tenant_namespace as tenant_namespace_module
from src.vector_store.tenant_namespace import (
    prefix_for_tenant,
    reset_prefix_cache,
    resolve_namespace_prefix,
)
from src.vector_store.turbopuffer import TurbopufferVectorStore


@pytest.fixture(autouse=True)
def _reset_prefix_cache_between_tests() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """The module-level prefix cache must never leak a resolved value across tests."""
    reset_prefix_cache()
    yield
    reset_prefix_cache()


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> TurbopufferVectorStore:
    """A real VectorStore whose get_vector_namespace runs unmodified -- no network I/O."""
    monkeypatch.setattr(settings.VECTOR_STORE, "TURBOPUFFER_API_KEY", "test-key")
    monkeypatch.setattr(settings.VECTOR_STORE, "TURBOPUFFER_REGION", "gcp-us-east4")
    return TurbopufferVectorStore()


async def _create_tenant(
    db_session: AsyncSession, *, vector_correlation_id: str | None
) -> str:
    """Insert and commit a tenant row, returning its freshly generated id."""
    tenant_id = str(generate_nanoid())
    db_session.add(
        models.Tenant(tenant_id=tenant_id, vector_correlation_id=vector_correlation_id)
    )
    await db_session.commit()
    return tenant_id


@pytest.mark.asyncio
async def test_flag_off_two_tenants_collide_on_the_same_namespace(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    store: TurbopufferVectorStore,
) -> None:
    """Today's behavior, reproduced: with the flag off, two tenants with identical
    workspace/peer inputs land on byte-identical namespaces."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    tenant_a = await _create_tenant(db_session, vector_correlation_id="tenant-a-app")
    tenant_b = await _create_tenant(db_session, vector_correlation_id="tenant-b-app")

    token = tenant_context.set(tenant_a)
    try:
        namespace_a = await store.get_vector_namespace("message", "default")
    finally:
        tenant_context.reset(token)

    token = tenant_context.set(tenant_b)
    try:
        namespace_b = await store.get_vector_namespace("message", "default")
    finally:
        tenant_context.reset(token)

    assert namespace_a == namespace_b


@pytest.mark.asyncio
async def test_flag_on_two_tenants_get_distinct_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    store: TurbopufferVectorStore,
) -> None:
    """Under the flag, the same workspace/peer inputs resolve to different namespaces
    per tenant, and neither equals the other's."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_a = await _create_tenant(db_session, vector_correlation_id="tenant-a-app")
    tenant_b = await _create_tenant(db_session, vector_correlation_id="tenant-b-app")

    token = tenant_context.set(tenant_a)
    try:
        namespace_a = await store.get_vector_namespace("message", "default")
    finally:
        tenant_context.reset(token)

    token = tenant_context.set(tenant_b)
    try:
        namespace_b = await store.get_vector_namespace("message", "default")
    finally:
        tenant_context.reset(token)

    assert namespace_a != namespace_b


@pytest.mark.asyncio
async def test_historical_correlation_id_reproduces_the_pre_tenancy_namespace(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    store: TurbopufferVectorStore,
) -> None:
    """A tenant keyed by its old app name gets exactly the namespace a pre-tenancy,
    single-instance deployment with that NAMESPACE would have produced -- the
    no-re-embed receipt."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_id = await _create_tenant(db_session, vector_correlation_id="hch-old-app")

    token = tenant_context.set(tenant_id)
    try:
        message_namespace = await store.get_vector_namespace("message", "default")
        document_namespace = await store.get_vector_namespace(
            "document", "default", observer="alice", observed="bob"
        )
    finally:
        tenant_context.reset(token)

    assert (
        message_namespace == f"hch-old-app.msg.{_hash_namespace_components('default')}"
    )
    assert document_namespace == (
        f"hch-old-app.doc.{_hash_namespace_components('default', 'alice', 'bob')}"
    )


@pytest.mark.asyncio
async def test_flag_off_uses_the_configured_namespace_even_with_explicit_prefix(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    """With the flag off, settings.VECTOR_STORE.NAMESPACE wins over both a bound
    tenant and an explicit prefix -- what keeps single-tenant output byte-identical."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings.VECTOR_STORE, "NAMESPACE", "single-tenant-namespace")
    tenant_id = await _create_tenant(
        db_session, vector_correlation_id="should-never-be-read"
    )

    token = tenant_context.set(tenant_id)
    try:
        resolved = await resolve_namespace_prefix("explicit-prefix-should-also-lose")
    finally:
        tenant_context.reset(token)

    assert resolved == "single-tenant-namespace"


@pytest.mark.asyncio
async def test_flag_on_no_tenant_bound_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the flag, resolving with no ambient tenant raises rather than guessing."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(VectorNamespaceUnresolved):
            await resolve_namespace_prefix()
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_flag_on_bound_tenant_with_no_registered_row_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant_id bound in context but absent from the tenants table raises rather
    than falling back to using that id as the prefix."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    unregistered_tenant_id = f"unregistered-{generate_nanoid()}"
    token = tenant_context.set(unregistered_tenant_id)
    try:
        with pytest.raises(VectorNamespaceUnresolved):
            await resolve_namespace_prefix()
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_flag_on_tenant_with_unset_correlation_id_fails_closed(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    """A registered tenant with no vector_correlation_id raises rather than silently
    falling back to its tenant id, which would orphan an existing corpus."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_id = await _create_tenant(db_session, vector_correlation_id=None)

    token = tenant_context.set(tenant_id)
    try:
        with pytest.raises(VectorNamespaceUnresolved):
            await resolve_namespace_prefix()
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_flag_on_explicit_prefix_overrides_the_ambient_tenant(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    """The background-path contract: an explicit prefix wins over whatever tenant
    happens to be ambient."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_id = await _create_tenant(
        db_session, vector_correlation_id="ambient-tenant-app"
    )

    token = tenant_context.set(tenant_id)
    try:
        resolved = await resolve_namespace_prefix("explicit-background-prefix")
    finally:
        tenant_context.reset(token)

    assert resolved == "explicit-background-prefix"


@pytest.mark.asyncio
async def test_prefix_for_tenant_fails_closed_on_missing_tenant_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A background row with no tenant id (None or empty) must raise, flag on."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    with pytest.raises(VectorNamespaceUnresolved):
        await prefix_for_tenant(None)
    with pytest.raises(VectorNamespaceUnresolved):
        await prefix_for_tenant("")


@pytest.mark.asyncio
async def test_prefix_for_tenant_reads_the_database_only_once_per_tenant(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    """A second resolution for the same tenant is served from the process cache --
    the loader is consulted exactly once."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    tenant_id = await _create_tenant(db_session, vector_correlation_id="cached-app")

    real_load_prefix = tenant_namespace_module._load_prefix  # pyright: ignore[reportPrivateUsage]
    load_count = 0

    async def counting_load_prefix(tid: str) -> str:
        nonlocal load_count
        load_count += 1
        return await real_load_prefix(tid)

    monkeypatch.setattr(tenant_namespace_module, "_load_prefix", counting_load_prefix)

    first = await prefix_for_tenant(tenant_id)
    second = await prefix_for_tenant(tenant_id)

    assert first == second == "cached-app"
    assert load_count == 1
