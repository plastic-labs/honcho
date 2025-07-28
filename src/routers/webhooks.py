import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path
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
    Set the webhook endpoint URL and enable specified events.
    """
    print(jwt_params)
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")
    print("after auth")
    return await crud.create_webhook_endpoint(db, workspace_id, webhook)


@router.put("", response_model=schemas.WebhookEndpoint)
async def update_webhook_endpoint(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook_update: schemas.WebhookEndpointUpdate = Body(
        ..., description="New webhook URL"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Update the webhook endpoint URL.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.update_webhook_endpoint(db, workspace_id, webhook_update.url)


@router.get("", response_model=schemas.WebhookEndpoint)
async def get_webhook_configuration(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get the webhook configuration for the workspace.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.get_webhook_configuration(db, workspace_id)


@router.delete("", response_model=None)
async def delete_webhook(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Delete the webhook endpoint and disable all events.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    await crud.delete_webhook_endpoint(db, workspace_id)


@router.post("/events", response_model=schemas.Webhook)
async def add_webhook_event(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook: schemas.WebhookCreate = Body(
        ..., description="Webhook event subscription"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Subscribe to a webhook event.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.add_webhook_event(db, workspace_id, webhook)


@router.put("/events", response_model=schemas.Webhook)
async def update_event_status(
    workspace_id: Annotated[str, Path(description="Workspace ID")],
    webhook_update: schemas.WebhookUpdate = Body(
        ..., description="Webhook event subscription"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
) -> schemas.Webhook:
    """
    Enable or disable a specific webhook event subscription.
    """
    if not jwt_params.ad and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.update_webhook_status(db, workspace_id, webhook_update)
