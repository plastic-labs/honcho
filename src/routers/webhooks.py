import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.crud import webhook as webhook_crud
from src.dependencies import db
from src.exceptions import AuthenticationException
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/webhooks",
    tags=["webhooks"],
)


@router.post("", response_model=schemas.WebhookEndpoint)
async def create_webhook_endpoint(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook: schemas.WebhookEndpointCreate = Body(
        ..., description="Webhook endpoint parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Create a new webhook endpoint URL for the workspace.
    All events will be automatically sent to this endpoint.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")
    return await webhook_crud.create_webhook_endpoint(db, workspace_id, webhook)


@router.get("", response_model=list[schemas.WebhookEndpoint])
async def list_webhook_endpoints(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    List all webhook endpoints for the workspace.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await webhook_crud.list_webhook_endpoints(db, workspace_id)


@router.delete("/{endpoint_id}", response_model=None)
async def delete_webhook_endpoint(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    endpoint_id: Annotated[str, Path(description="Webhook endpoint ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Delete a specific webhook endpoint.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    await webhook_crud.delete_webhook_endpoint(db, workspace_id, endpoint_id)
