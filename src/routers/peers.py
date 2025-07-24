import logging
from collections.abc import AsyncGenerator

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Path,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import apaginate
from mirascope.llm import Stream
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas
from src.dependencies import db
from src.dialectic import chat as dialectic_chat
from src.exceptions import AuthenticationException, ResourceNotFoundException
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
    options: schemas.PeerGet | None = Body(
        None, description="Filtering options for the peers list"
    ),
    db: AsyncSession = db,
):
    """Get All Peers for a Workspace"""
    filter_param = None
    if options and hasattr(options, "filter"):
        filter_param = options.filter
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_peers(workspace_name=workspace_id, filters=filter_param),
    )


@router.post(
    "",
    response_model=schemas.Peer,
)
async def get_or_create_peer(
    workspace_id: str = Path(..., description="ID of the workspace"),
    peer: schemas.PeerCreate = Body(..., description="Peer creation parameters"),
    jwt_params: JWTParams = Depends(require_auth()),
    db: AsyncSession = db,
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
    db: AsyncSession = db,
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
    options: schemas.SessionGet | None = Body(
        None, description="Filtering options for the sessions list"
    ),
    db: AsyncSession = db,
):
    """Get All Sessions for a Peer"""
    filter_param = None

    if options and hasattr(options, "filter"):
        filter_param = options.filter
        if filter_param == {}:
            filter_param = None

    return await apaginate(
        db,
        await crud.get_sessions_for_peer(
            workspace_name=workspace_id,
            peer_name=peer_id,
            filters=filter_param,
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
    db: AsyncSession = db,
):
    # Get or create the peer to ensure it exists
    await crud.get_or_create_peers(
        db, workspace_name=workspace_id, peers=[schemas.PeerCreate(name=peer_id)]
    )

    if not options.stream:
        response = await dialectic_chat(
            workspace_name=workspace_id,
            peer_name=peer_id,
            target_name=options.target,
            session_name=options.session_id,
            query=options.query,
            stream=options.stream,
        )
        return schemas.DialecticResponse(content=str(response))

    async def parse_stream() -> AsyncGenerator[str, None]:
        try:
            stream = await dialectic_chat(
                workspace_name=workspace_id,
                peer_name=peer_id,
                target_name=options.target,
                session_name=options.session_id,
                query=options.query,
                stream=options.stream,
            )
            if isinstance(stream, Stream):
                async for chunk, _ in stream:
                    yield chunk.content
            else:
                raise HTTPException(status_code=500, detail="Invalid stream type")
        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return StreamingResponse(
        content=parse_stream(), media_type="text/event-stream", status_code=200
    )


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
    db: AsyncSession = db,
):
    """Get a peer's working representation for a session.

    If a session_id is provided in the body, we get the working representation of the peer in that session.
    If a target is provided, we get the representation of the target from the perspective of the peer.
    If no target is provided, we get the global representation of the peer.
    """
    try:
        # If no target specified, get global representation (peer observing themselves)
        target_peer = options.target if options.target is not None else peer_id

        representation = await crud.get_working_representation(
            db, workspace_id, peer_id, target_peer, options.session_id
        )
        return {"representation": representation}
    except ValueError as e:
        logger.warning(
            f"Failed to get working representation for peer {peer_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Peer or session not found") from e


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
    search: schemas.MessageSearchOptions = Body(
        ..., description="Message search parameters "
    ),
    db: AsyncSession = db,
):
    """Search a Peer"""
    stmt = await crud.search(
        search.query,
        workspace_name=workspace_id,
        peer_name=peer_id,
        semantic=search.semantic,
    )

    return await apaginate(db, stmt)
