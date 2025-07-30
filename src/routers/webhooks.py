import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.crud import webhook as crud
from src.dependencies import db
from src.exceptions import AuthenticationException
from src.security import JWTParams, require_auth
from src.webhooks.event_emitter import webhook_emitter
from src.webhooks.events import QueueEmptyPayload

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/webhooks",
    tags=["webhooks"],
)


@router.post("", response_model=schemas.WebhookEndpoint)
async def get_or_create_webhook_endpoint(
    webhook: schemas.WebhookEndpointCreate = Body(
        ..., description="Webhook endpoint parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Get or create a webhook endpoint URL.
    """
    if (
        not jwt_params.ad
        and jwt_params.w is not None
        and jwt_params.w != webhook.workspace_name
    ):
        raise AuthenticationException("Unauthorized access to resource")

    return await crud.get_or_create_webhook_endpoint(db, webhook)


@router.get("", response_model=Page[schemas.WebhookEndpoint])
async def list_webhook_endpoints(
    workspace_id: str,
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    List all webhook endpoints, optionally filtered by workspace.
    """
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    stmt = await crud.list_webhook_endpoints(db, workspace_id)
    return await apaginate(db, stmt)


@router.delete("/{endpoint_id}", response_model=None)
async def delete_webhook_endpoint(
    endpoint_id: Annotated[str, Path(description="Webhook endpoint ID")],
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
):
    """
    Delete a specific webhook endpoint.
    """

    if not jwt_params.ad:
        raise AuthenticationException("Unauthorized access to resource")

    await crud.delete_webhook_endpoint(db, endpoint_id)


@router.post("/test-emit", response_model=None)
async def test_emit(
    workspace_name: str | None = None,
):
    """
    Test webhook emission, optionally for a specific workspace.
    """
    await webhook_emitter.emit_queue_empty(
        workspace_name or "global",
        QueueEmptyPayload(
            workspace_name=workspace_name or "global",
            queue_type="summary",
        ),
    )
