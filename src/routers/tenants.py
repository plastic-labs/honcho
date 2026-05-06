"""Tenant management router for multi-tenant Honcho deployments."""

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.dependencies import db
from src.models import Tenant as TenantModel
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tenants",
    tags=["tenants"],
)


@router.post("", response_model=schemas.Tenant)
async def create_tenant(
    response: Response,
    tenant: schemas.TenantCreate = Body(...),
    jwt_params: JWTParams = Depends(require_auth(admin=True)),
    db: AsyncSession = db,
):
    """Create a new tenant (admin-only)."""
    existing = await db.scalar(
        select(TenantModel).where(TenantModel.name == tenant.name)
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Tenant '{tenant.name}' already exists",
        )

    new_tenant = TenantModel(
        name=tenant.name,
        h_metadata=tenant.metadata,
    )
    db.add(new_tenant)
    await db.commit()
    await db.refresh(new_tenant)

    response.status_code = 201
    return new_tenant


@router.get(
    "/{tenant_id}",
    response_model=schemas.Tenant,
    dependencies=[Depends(require_auth(admin=True))],
)
async def get_tenant(
    tenant_id: str,
    db: AsyncSession = db,
):
    """Get a tenant by ID (admin-only)."""
    tenant = await db.scalar(
        select(TenantModel).where(TenantModel.id == tenant_id)
    )
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return tenant


@router.post(
    "/list",
    response_model=Page[schemas.Tenant],
    dependencies=[Depends(require_auth(admin=True))],
)
async def list_tenants(
    db: AsyncSession = db,
):
    """List all tenants (admin-only)."""
    return await apaginate(db, select(TenantModel))


@router.delete(
    "/{tenant_id}",
    status_code=204,
    dependencies=[Depends(require_auth(admin=True))],
)
async def delete_tenant(
    tenant_id: str,
    db: AsyncSession = db,
):
    """Delete a tenant and all associated resources (admin-only)."""
    tenant = await db.scalar(
        select(TenantModel).where(TenantModel.id == tenant_id)
    )
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    await db.delete(tenant)
    await db.commit()
