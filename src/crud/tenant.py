"""CRUD for the tenant registry — the above-tenant provisioning surface.

Callers hold a service session (``service_db``): the registry sits above
row-level security by design, and ``tracked_db`` would fail closed because no
tenant is bound while the tenant is being created.
"""

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.exceptions import ConflictException, ResourceNotFoundException
from src.utils.types import GetOrCreateResult

logger = logging.getLogger(__name__)


async def get_tenant(db: AsyncSession, tenant_id: str) -> models.Tenant:
    """Fetch a tenant row or raise 404."""
    tenant = await db.get(models.Tenant, tenant_id)
    if tenant is None:
        raise ResourceNotFoundException(f"Tenant {tenant_id} not found")
    return tenant


async def get_or_create_tenant(
    db: AsyncSession,
    tenant_id: str,
    vector_correlation_id: str | None,
    tier: str,
) -> GetOrCreateResult[models.Tenant]:
    """Idempotent create.

    A retry with identical fields returns the existing row; the same
    tenant_id with different fields is a conflict — this API never mutates
    an existing tenant (changing a tenant's fields is out of scope here).
    """

    def _matching_or_conflict(existing: models.Tenant) -> models.Tenant:
        if (
            existing.vector_correlation_id == vector_correlation_id
            and existing.tier == tier
        ):
            return existing
        raise ConflictException(
            f"Tenant {tenant_id} already exists with different fields"
        )

    existing = await db.get(models.Tenant, tenant_id)
    if existing is not None:
        return GetOrCreateResult(resource=_matching_or_conflict(existing), created=False)

    tenant = models.Tenant(
        tenant_id=tenant_id,
        vector_correlation_id=vector_correlation_id,
        tier=tier,
    )
    db.add(tenant)
    try:
        await db.commit()
    except IntegrityError:
        # Lost the create race to a concurrent retry of the same provisioning
        # call — apply the same idempotency contract to the winner's row.
        await db.rollback()
        existing = await db.get(models.Tenant, tenant_id)
        if existing is None:  # pragma: no cover - delete raced the retry
            # region ai
            # Reachable only when a DELETE of this tenant_id commits between
            # our failed INSERT and this re-read. A typed 409 keeps the
            # caller's retry loop alive; re-raising the IntegrityError would
            # surface as an opaque 500.
            # endregion
            raise ConflictException(
                f"Tenant {tenant_id} creation raced a concurrent delete; retry"
            ) from None
        return GetOrCreateResult(resource=_matching_or_conflict(existing), created=False)
    logger.info("Created tenant %s (tier=%s)", tenant_id, tier)
    return GetOrCreateResult(resource=tenant, created=True)


async def delete_tenant(db: AsyncSession, tenant_id: str) -> None:
    """Delete an EMPTY tenant (the provisioning-rollback primitive)."""
    # region ai
    # Not eviction: every tenant-scoped table FKs tenants.tenant_id with no
    # ON DELETE action, so Postgres refuses the delete while dependent rows
    # exist and we surface that as a 409. Cascading the delete of a tenant
    # that still owns data is deliberately out of scope for this endpoint;
    # removing a populated tenant is an operational action performed through
    # other tooling, never this API.
    # endregion
    tenant = await get_tenant(db, tenant_id)
    await db.delete(tenant)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise ConflictException(
            f"Tenant {tenant_id} is not empty; only tenants without data can "
            + "be deleted here"
        ) from exc
    logger.info("Deleted tenant %s", tenant_id)
