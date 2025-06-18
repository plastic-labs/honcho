import logging
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Path,
    Query,
)
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
)
from src.routers.messages import enqueue
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/peers",
    tags=["peers"],
)


@router.post(
    "/list",
    response_model=Page[schemas.Peer],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: Optional[schemas.PeerGet] = Body(
        None, description="Filtering options for the peers list"
    ),
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
        await crud.get_peers(workspace_name=workspace_id, filter=filter_param),
    )


@router.post(
    "",
    response_model=schemas.Peer,
)
async def get_or_create_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer: schemas.PeerCreate = Body(..., description="Peer creation parameters"),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a Peer by ID

    If peer_id is provided as a query parameter, it uses that (must match JWT workspace_id).
    Otherwise, it uses the peer_id from the JWT token.
    """
    # validate workspace query param
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    if peer.name:
        if not jwt_params.ad and jwt_params.p is not None and jwt_params.p != peer.name:
            raise AuthenticationException("Unauthorized access to resource")
    else:
        # Use peer_id from JWT
        if not jwt_params.p:
            raise AuthenticationException("Peer ID not found in query parameter or JWT")
        peer.name = jwt_params.p
    peer = (
        await crud.get_or_create_peers(db, workspace_name=workspace_id, peers=[peer])
    )[0]
    return peer


@router.put(
    "/{peer_id}",
    response_model=schemas.Peer,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def update_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer to update"),
    peer: schemas.PeerUpdate = Body(..., description="Updated peer parameters"),
    db=db,
):
    """Update a Peer's name and/or metadata"""
    updated_peer = await crud.update_peer(
        db, workspace_name=workspace_id, peer_name=peer_id, peer=peer
    )
    return updated_peer


@router.post(
    "/{peer_id}/sessions",
    response_model=Page[schemas.Session],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_sessions_for_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: Optional[schemas.SessionGet] = Body(
        None, description="Filtering options for the sessions list"
    ),
    db=db,
):
    """Get All Sessions for a Peer"""
    filter_param = None
    is_active = True

    if options:
        if hasattr(options, "filter"):
            filter_param = options.filter
            if filter_param == {}:
                filter_param = None
        if hasattr(options, "is_active"):
            is_active = options.is_active

    return await paginate(
        db,
        await crud.get_sessions_for_peer(
            workspace_name=workspace_id,
            peer_name=peer_id,
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
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def chat(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: schemas.DialecticOptions = Body(
        ..., description="Dialectic Endpoint Parameters"
    ),
):
    return schemas.DialecticResponse(
        content=f"Hello, {peer_id}! You are chatting with {options.target} in {options.session_id} in workspace {workspace_id} with options {options}",
    )


@router.post(
    "/{peer_id}/messages",
    response_model=list[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
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
    workspace_name, peer_name = workspace_id, peer_id
    """Bulk create messages for a peer while maintaining order."""
    try:
        created_messages = await crud.create_messages_for_peer(
            db,
            messages=messages.messages,
            workspace_name=workspace_name,
            peer_name=peer_name,
        )

        # Create payloads for all messages
        payloads = [
            {
                "workspace_name": workspace_name,
                "session_name": None,
                "message_id": message.id,
                "content": message.content,
                "metadata": message.h_metadata,
                "peer_name": message.peer_name,
            }
            for message in created_messages
        ]

        # Enqueue all messages in one call
        background_tasks.add_task(enqueue, payloads)  # type: ignore
        logger.info(
            f"Batch of {len(created_messages)} messages created and queued for processing"
        )

        return created_messages
    except ValueError as e:
        logger.error(f"Failed to create batch messages for peer {peer_id}: {str(e)}")
        raise ResourceNotFoundException("Peer not found") from e


@router.post(
    "/{peer_id}/messages/list",
    response_model=Page[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_messages_for_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: Optional[schemas.MessageGet] = Body(
        None, description="Filtering options for the messages list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    db=db,
):
    """Get all messages for a peer"""
    try:
        filter = None
        if options and hasattr(options, "filter"):
            filter = options.filter
            if filter == {}:
                filter = None

        messages_query = await crud.get_messages_for_peer(
            workspace_name=workspace_id,
            peer_name=peer_id,
            filter=filter,
            reverse=reverse,
        )

        return await paginate(db, messages_query)
    except ValueError as e:
        logger.warning(f"Failed to get messages for peer {peer_id}: {str(e)}")
        raise ResourceNotFoundException("Peer not found") from e


@router.post(
    "/{peer_id}/representation",
    response_model=dict[str, object],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_working_representation(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    options: schemas.PeerRepresentationGet = Body(
        ..., description="Options for getting the peer representation"
    ),
    db=db,
):
    """Get a peer's working representation for a session.

    If peer_id is provided in body, the representation is of that peer, from our perspective.
    """

    stub = {
        "final_observations": {
            "explicit": [
                {
                    "content": "User said: 'Hey Mel!' - addressing someone named Mel",
                    "created_at": "2023-05-08T13:56:00+00:00",
                },
                {
                    "content": "User said: 'Good to see you!' - expressing positive sentiment about seeing this person",
                    "created_at": "2023-05-08T13:56:00+00:00",
                },
                {
                    "content": "User said: 'How have you been?' - asking about the other person's recent state or experiences",
                    "created_at": "2023-05-08T13:56:00+00:00",
                },
            ],
            "deductive": [
                {
                    "conclusion": "The user believes they have encountered 'Mel' before",
                    "premises": [
                        "User said 'Good to see you!' which implies previous encounters"
                    ],
                    "created_at": "2023-05-08T13:56:00+00:00",
                },
                {
                    "conclusion": "The user is initiating a conversational exchange",
                    "premises": [
                        "User said 'Hey Mel!' as a greeting",
                        "User asked 'How have you been?' which is a conversation starter",
                    ],
                    "created_at": "2023-05-08T13:56:00+00:00",
                },
            ],
            "inductive": [],
            "abductive": [
                {
                    "conclusion": "The user believes they have an established relationship or familiarity with 'Mel'",
                    "premises": [
                        "Casual greeting format 'Hey Mel!'",
                        "Familiar expression 'Good to see you!'",
                        "Personal inquiry about wellbeing",
                    ],
                    "created_at": "2023-05-08T13:56:00+00:00",
                }
            ],
        },
    }

    return stub


@router.post(
    "/{peer_id}/search",
    response_model=Page[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def search_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    query: str = Body(..., description="Search query"),
    db=db,
):
    """Search a Peer"""
    stmt = await crud.search(query, workspace_name=workspace_id, peer_name=peer_id)

    return await paginate(db, stmt)


@router.get(
    "/{peer_id}/deriver/status",
    response_model=schemas.DeriverStatus,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", peer_name="peer_id"))
    ],
)
async def get_peer_deriver_status(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer_id: str = Path(..., description="ID of the peer"),
    session_id: Optional[str] = Query(None, description="Optional session ID to filter by"),
    include_sender: bool = Query(False, description="Include work units triggered by this peer"),
    db=db,
):
    """Get the deriver processing status for a peer, optionally scoped to a session"""
    try:
        return await crud.get_peer_deriver_status(
            db,
            workspace_name=workspace_id,
            peer_name=peer_id,
            session_name=session_id,
            include_sender=include_sender,
        )
    except ResourceNotFoundException as e:
        logger.warning(f"Failed to get deriver status for peer {peer_id}: {str(e)}")
        raise ResourceNotFoundException("Peer not found") from e
