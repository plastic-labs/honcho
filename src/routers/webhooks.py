import logging

from fastapi import APIRouter, Body, Depends, Path
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.config import settings
from src.crud import webhook as crud
from src.dependencies import db
from src.exceptions import AuthenticationException, ConflictException
from src.security import JWTParams, require_auth
from src.webhooks.events import (
    TestEvent,
    publish_webhook_event,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/webhooks",
    tags=["webhooks"],
)


@router.post("", response_model=schemas.WebhookEndpoint)
async def get_or_create_webhook_endpoint(
    workspace_id: str = Path(..., description="Workspace ID"),
    webhook: schemas.WebhookEndpointCreate = Body(
        ..., description="Webhook endpoint parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
) -> schemas.WebhookEndpoint:
    """
    Get or create a webhook endpoint URL.
    """
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    try:
        return await crud.get_or_create_webhook_endpoint(
            db, workspace_id, webhook=webhook
        )
    except ValueError as e:
        raise ConflictException(
            f"Maximum number of webhook endpoints ({settings.WEBHOOK.MAX_WORKSPACE_LIMIT}) reached for this workspace."
        ) from e


@router.get("", response_model=Page[schemas.WebhookEndpoint])
async def list_webhook_endpoints(
    workspace_id: str = Path(..., description="Workspace ID"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
) -> Page[schemas.WebhookEndpoint]:
    """
    List all webhook endpoints, optionally filtered by workspace.
    """
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    stmt = await crud.list_webhook_endpoints(workspace_id)
    return await apaginate(db, stmt)


@router.delete("/{endpoint_id}", response_model=None)
async def delete_webhook_endpoint(
    workspace_id: str = Path(..., description="Workspace ID"),
    endpoint_id: str = Path(..., description="Webhook endpoint ID"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
) -> None:
    """
    Delete a specific webhook endpoint.
    """

    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    await crud.delete_webhook_endpoint(db, workspace_id, endpoint_id)


@router.get("/test")
async def test_emit(
    workspace_id: str = Path(..., description="Workspace ID"),
    jwt_params: JWTParams = Depends(require_auth()),
) -> None:
    """
    Test publishing a webhook event.
    """
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unable to publish test webhook")

    event = TestEvent(workspace_id=workspace_id)
    await publish_webhook_event(event)
