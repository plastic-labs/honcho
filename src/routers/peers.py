import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from fastapi import BackgroundTasks

from src import crud, schemas
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
)
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/peers",
    tags=["peers"],
)


@router.post(
    "/list",
    response_model=Page[schemas.Peer],
    dependencies=[Depends(require_auth(app_id="workspace_id"))],
)
async def get_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: Optional[schemas.PeerGet] = Body(
        None, description="Filtering options for the peers list"
    ),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db=db,
):
    """Get All Peers for a Workspace"""
    filter_param = None
    if options and hasattr(options, "filter"):
        filter_param = options.filter
        if filter_param == {}:
            filter_param = None

    return await paginate(
        db,
        await crud.get_peers(workspace_name=workspace_id, reverse=reverse, filter=filter_param),
    )


@router.post(
    "",
    response_model=schemas.Peer,
)
async def get_or_create_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer: schemas.PeerCreate = Body(
        ..., description="Peer creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a Peer by ID

    If peer_id is provided as a query parameter, it uses that (must match JWT workspace_id).
    Otherwise, it uses the peer_id from the JWT token.
    """
    # validate workspace query param
    if not jwt_params.ad and jwt_params.ap is not None and jwt_params.ap != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    if peer.name:
        if not jwt_params.ad and jwt_params.us is not None and jwt_params.us != peer.name:
            raise AuthenticationException("Unauthorized access to resource")
    else:
        # Use peer_id from JWT
        if not jwt_params.us:
            raise AuthenticationException("Peer ID not found in query parameter or JWT")
        peer.name = jwt_params.us
    peer = await crud.get_or_create_peer(db, workspace_name=workspace_id, peer=peer)
    return peer


@router.put(
    "/{peer_id}",
    response_model=schemas.Peer,
    dependencies=[Depends(require_auth(app_id="workspace_id", user_id="peer_id"))],
)
async def update_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer to update"),
    peer: schemas.PeerUpdate = Body(..., description="Updated peer parameters"),
    db=db,
):
    """Update a Peer's name and/or metadata"""
    updated_peer = await crud.update_peer(db, workspace_name=workspace_id, peer_name=peer_id, peer=peer)
    return updated_peer


@router.post(
    "/{peer_id}/sessions",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(app_id="workspace_id", user_id="peer_id"))],
)
async def get_peer_sessions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: Optional[schemas.SessionGet] = Body(
        None, description="Filtering options for the sessions list"
    ),
    reverse: bool = Query(False, description="Whether to reverse the order of results"),
    db=db,
):
    """Get All Sessions for a Peer"""
    filter_param = None
    is_active = False  # Default from schemas
    
    if options:
        if hasattr(options, "filter"):
            filter_param = options.filter
            if filter_param == {}:
                filter_param = None
        if hasattr(options, "is_active"):
            is_active = options.is_active

    return await paginate(
        db,
        await crud.get_peer_sessions(
            workspace_name=workspace_id,
            peer_name=peer_id,
            reverse=reverse,
            is_active=is_active,
            filter=filter_param,
        ),
    )


@router.post(
    "/{peer_id}/chat",
    response_model=schemas.DialecticResponse,
    responses={
        200: {
            "description": "Response to a question informed by Honcho's User Representation",
            "content": {"text/event-stream": {}},
        },
    },
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", user_id="peer_id")
        )
    ],
)
async def chat(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: schemas.DialecticOptions = Body(
        ..., description="Dialectic Endpoint Parameters"
    ),
):
    pass

@router.post(
    "/{peer_id}/messages",
    response_model=list[schemas.Message],
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", user_id="peer_id")
        )
    ],
)
async def create_messages_for_peer(
    background_tasks: BackgroundTasks,
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    messages: schemas.MessageBatchCreate = Body(
        ..., description="Batch of messages to create"
    ),
    db=db,
):
    """Create messages for a peer"""
    pass
