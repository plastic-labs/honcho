"""Tenant registry API — the above-tenant provisioning surface.

The control plane creates a tenant here before any tenant-scoped credential
or write can exist. Authentication is a service secret, not a JWT — see
``require_tenant_api``.
"""

import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Response

from src import schemas
from src.config import settings
from src.crud import tenant as tenant_crud
from src.dependencies import service_db
from src.exceptions import AuthenticationException, DisabledException, ValidationException
from src.models import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)


async def require_tenant_api(
    x_tenant_api_key: Annotated[str | None, Header()] = None,
) -> None:
    # region ai
    # The above-tenant auth plane. This router cannot use require_auth: under
    # MULTI_TENANT every JWT must carry a tenant claim, and at creation time the
    # tenant does not exist to be claimed. Fail-closed twice over — the API is
    # disabled unless MULTI_TENANT is on AND the secret is configured, and the
    # header must match in constant time. The two planes never mix: JWTs
    # authenticate within a tenant; this secret authenticates the registry.
    # Compared as bytes: compare_digest rejects non-ASCII str (a raw high-byte
    # header would 500 instead of 401), while bytes have no such restriction.
    # endregion
    if not settings.MULTI_TENANT or not settings.TENANT_API.SECRET:
        raise DisabledException(
            "The tenant API is disabled: it requires MULTI_TENANT and a "
            + "configured TENANT_API_SECRET"
        )
    if not x_tenant_api_key or not hmac.compare_digest(
        x_tenant_api_key.encode("utf-8"), settings.TENANT_API.SECRET.encode("utf-8")
    ):
        logger.warning("Tenant API request rejected: invalid or missing key")
        raise AuthenticationException("Invalid tenant API key")


router = APIRouter(
    prefix="/tenants",
    tags=["tenants"],
    dependencies=[Depends(require_tenant_api)],
)


@router.post("", response_model=schemas.Tenant)
async def create_tenant(body: schemas.TenantCreate, response: Response):
    """Idempotently create a tenant: 201 created, 200 already-exists-identical,
    409 exists-with-different-fields."""
    async with service_db("tenants.create") as db:
        result = await tenant_crud.get_or_create_tenant(
            db,
            tenant_id=body.tenant_id,
            vector_correlation_id=body.vector_correlation_id,
            tier=body.tier,
        )
        response.status_code = 201 if result.created else 200
        return result.resource


@router.get("/{tenant_id}", response_model=schemas.Tenant)
async def get_tenant(tenant_id: Annotated[str, Path()]):
    """Fetch a tenant row (the control plane's reconciliation read)."""
    async with service_db("tenants.get", read_only=True) as db:
        return await tenant_crud.get_tenant(db, tenant_id)


@router.delete("/{tenant_id}", status_code=204)
async def delete_tenant(tenant_id: Annotated[str, Path()]) -> None:
    """Delete an empty tenant (provisioning rollback): 409 if it has data."""
    if tenant_id == DEFAULT_TENANT_ID:
        raise ValidationException(
            f"The {DEFAULT_TENANT_ID!r} tenant is the single-tenant bootstrap "
            + "row and cannot be deleted"
        )
    async with service_db("tenants.delete") as db:
        await tenant_crud.delete_tenant(db, tenant_id)
