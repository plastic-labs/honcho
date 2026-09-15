"""Ambient tenant identity for telemetry surfaces.

Under ``MULTI_TENANT`` the request's tenant (API, bound in ``src.security`` from the JWT
``tn`` claim) or the work unit's tenant (deriver, bound in ``src.deriver.queue_manager``
from the work-unit key) already lives in ``src.db.tenant_context`` for the whole scope.
Telemetry reads it here rather than threading a tenant through every event class and
emit site.

Flag-off this returns ``None`` unconditionally, even if something set the ContextVar:
a single-tenant instance's telemetry must stay byte-identical to pre-tenancy output,
where the ``namespace`` (the instance) is the tenant.
"""

from src.config import settings

# Event categories that have no single tenant by construction: the reconciler runs
# across tenants on the service role, so its events carry the instance only. An
# untenanted emit in one of these categories is normal, not a bug.
TENANTLESS_CATEGORIES: frozenset[str] = frozenset({"reconciliation"})


def current_tenant_id() -> str | None:
    """The bound tenant under ``MULTI_TENANT``; ``None`` flag-off or when nothing is bound."""
    if not settings.MULTI_TENANT:
        return None
    # Lazy: src.db imports src.telemetry.prometheus.metrics, which imports this module.
    from src.db import tenant_context

    return tenant_context.get()
