"""The vector-store namespace prefix, resolved per tenant.

A namespace is ``{prefix}.{doc|msg}.{hash(workspace[, observer, observed])}``. The hash
carries no tenant, so the prefix is the only thing keeping one tenant's vectors apart
from another's. Single-tenant, that prefix is the instance's own ``VECTOR_STORE.NAMESPACE``
and the scheme is already tenant-pure. On an instance serving many tenants it is not, so
under ``MULTI_TENANT`` the prefix comes from the tenant instead.
"""

# region ai
# The value is each tenant's vector_correlation_id: for a tenant that once had its own
# instance this is its historical app name, which is what its existing namespaces are
# keyed by, so preserving it reproduces every namespace name exactly and no vector has to
# move. The control plane is expected to set it for every tenant it registers, passing
# the tenant id for a tenant with no history — nothing in this repo enforces that, so a
# missing key is a provisioning bug rather than a supported state. This refuses rather
# than guessing: falling back to the tenant id would hand a migrated tenant a brand-new
# namespace and quietly orphan everything it had, surfacing as an empty search rather
# than an error.
# endregion

from src.config import settings
from src.exceptions import VectorNamespaceUnresolved, VectorStoreError

# region ai
# Resolved lazily at the vector call rather than eagerly where the tenant is bound: the
# auth boundary has no database session (`src/security.py` auth_dependency takes only the
# request and credentials), so resolving there would open one on every authenticated
# request, including the large majority that never touch the vector store. Cached for the
# life of the process because the registry refuses to mutate an existing tenant, so the
# read happens once per tenant per process and the session-free read path stays
# session-free on every warm call.
#
# The one way a cached value goes stale: deleting an empty tenant and recreating the same
# id with a different key. There is no eviction for that — a warm process keeps the old
# prefix until it restarts — and no in-process fix would be complete anyway, since every
# replica caches independently. Call reset_prefix_cache() if it ever happens.
# endregion
_prefix_cache: dict[str, str] = {}


async def resolve_namespace_prefix(explicit: str | None = None) -> str:
    """The namespace prefix for the current tenant.

    Args:
        explicit: Use this prefix instead of the ambient tenant's. The cross-tenant
            background paths (the reconciler, immediate embedding, the soft-delete
            sweep) pass the prefix belonging to the row they are working on, because
            they legitimately serve many tenants per cycle and have no ambient one.

    Returns:
        The instance's configured namespace when ``MULTI_TENANT`` is off — taking
        precedence over ``explicit``, so single-tenant output is byte-identical to
        pre-tenancy behavior whoever calls it. Otherwise ``explicit``, else the
        bound tenant's key.

    Raises:
        VectorNamespaceUnresolved: ``MULTI_TENANT`` is on and either no tenant is
            bound or the bound tenant has no key.
        VectorStoreError: the key could not be read (a transient database failure).
    """
    # ai: flag-off wins over an explicit prefix — single-tenant has exactly one namespace, and this is what keeps that output byte-identical
    if not settings.MULTI_TENANT:
        return settings.VECTOR_STORE.NAMESPACE
    if explicit is not None:
        return explicit

    # ai: deferred so importing src.vector_store stays cheap — callers import it lazily to skip its dependencies when no external store is configured. There is no import cycle here, unlike src/telemetry/tenant.py, whose identical-looking deferral does close one
    from src.db import tenant_context

    tenant_id = tenant_context.get()
    if not tenant_id:
        raise VectorNamespaceUnresolved(
            "No tenant is bound for this vector-store call. Request paths bind one at "
            + "the auth boundary and the deriver binds one per work unit; a genuinely "
            + "cross-tenant path must pass an explicit prefix for the row it is handling."
        )

    return await prefix_for_tenant(tenant_id)


async def prefix_for_tenant(tenant_id: str | None) -> str:
    """The namespace prefix for a named tenant.

    For the cross-tenant background paths, which have no ambient tenant and work a row
    at a time: each row carries its own ``tenant_id``, so each gets its own namespace.

    Raises:
        VectorNamespaceUnresolved: ``MULTI_TENANT`` is on and the row has no tenant,
            or that tenant has no key.
        VectorStoreError: the key could not be read (a transient database failure).
    """
    if not settings.MULTI_TENANT:
        return settings.VECTOR_STORE.NAMESPACE
    if not tenant_id:
        raise VectorNamespaceUnresolved(
            "A row reached the vector store with no tenant while MULTI_TENANT is on; "
            + "its namespace cannot be resolved."
        )

    cached = _prefix_cache.get(tenant_id)
    if cached is not None:
        return cached

    prefix = await _load_prefix(tenant_id)
    _prefix_cache[tenant_id] = prefix
    return prefix


async def _load_prefix(tenant_id: str) -> str:
    """Read one tenant's key. Cross-tenant by nature, so it runs on the service role."""
    # ai: deferred for the same reason as above — keep src.vector_store cheap to import
    from src import models
    from src.dependencies import service_db

    # region ai
    # A failed read is transient and belongs to the retryable family, unlike the two
    # raises below. Left unnormalized it would surface as a raw database error at the
    # same point inside a cross-tenant batch and escape the per-row containment the
    # callers apply, taking every co-batched tenant down with it.
    # endregion
    try:
        async with service_db("vector_namespace_prefix", read_only=True) as db:
            tenant = await db.get(models.Tenant, tenant_id)
    except Exception as exc:
        raise VectorStoreError(
            f"Could not read the vector namespace key for tenant {tenant_id}."
        ) from exc

    if tenant is None:
        raise VectorNamespaceUnresolved(
            f"Tenant {tenant_id} is bound but not registered, so its vector namespace "
            + "cannot be resolved."
        )
    if not tenant.vector_correlation_id:
        raise VectorNamespaceUnresolved(
            f"Tenant {tenant_id} has no vector_correlation_id. The control plane sets "
            + "one for every tenant it registers — a tenant's historical app name if it "
            + "had its own instance, its tenant id otherwise. Refusing rather than "
            + "guessing: guessing would orphan an existing corpus behind an empty search."
        )
    return tenant.vector_correlation_id


def reset_prefix_cache() -> None:
    """Drop every cached key. For tests, and for a process that must re-read them."""
    _prefix_cache.clear()
