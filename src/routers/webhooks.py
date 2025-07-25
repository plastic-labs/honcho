import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/webhooks",
    tags=["webhooks"],
)


@router.post("", response_model=schemas.Webhook)
async def create_webhook(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook: schemas.WebhookCreate = Body(
        ..., description="Webhook creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Create a new webhook for the workspace.
    """
    # Check workspace access
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.create_webhook(db, workspace_id, webhook)


@router.get("", response_model=Page[schemas.Webhook])
async def get_webhooks(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    active_only: bool = Query(False, description="Only return active webhooks"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get all webhooks for the workspace.
    """
    # Check workspace access
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    stmt = await crud.get_all_webhooks(workspace_id, active_only=active_only)
    return await apaginate(db, stmt)


@router.get("/{webhook_id}", response_model=schemas.Webhook)
async def get_webhook(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook_id: Annotated[str, Path(description="Webhook ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get a specific webhook by ID.
    """
    # Check workspace access
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.get_webhook(db, workspace_id, webhook_id)


@router.patch("/{webhook_id}", response_model=schemas.Webhook)
async def update_webhook(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook_id: Annotated[str, Path(description="Webhook ID")],
    webhook: schemas.WebhookUpdate = Body(..., description="Webhook update parameters"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Update an existing webhook.
    """
    # Check workspace access
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.update_webhook(db, workspace_id, webhook_id, webhook)


@router.delete("/{webhook_id}")
async def delete_webhook(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook_id: Annotated[str, Path(description="Webhook ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Delete a webhook.
    """
    # Check workspace access
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    await crud.delete_webhook(db, workspace_id, webhook_id)
    return {"message": "Webhook deleted successfully"}
