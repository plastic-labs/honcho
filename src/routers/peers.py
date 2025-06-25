import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Path,
    Query,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from mirascope.llm import Stream
from sqlalchemy.ext.asyncio import AsyncSession

from src import agent, crud, schemas
from src.config import settings
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    DisabledException,
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
    db: AsyncSession = db,
):
    # Get or create the peer to ensure it exists
    await crud.get_or_create_peers(
        db, workspace_name=workspace_id, peers=[schemas.PeerCreate(name=peer_id)]
    )

    if not options.stream:
        return await agent.chat(
            workspace_id, peer_id, options.session_id, options.queries, options.stream
        )

    async def parse_stream():
        try:
            stream = await agent.chat(
                workspace_id,
                peer_id,
                options.session_id,
                options.queries,
                stream=True,
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
    db: AsyncSession = db,
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
    options: schemas.MessageGet | None = Body(
        None, description="Filtering options for the messages list"
    ),
    reverse: bool | None = Query(
        False, description="Whether to reverse the order of results"
    ),
    db: AsyncSession = db,
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
    db: AsyncSession = db,
):
    """Get a peer's working representation for a session.

    If a session_id is provided in the body, we get the working representation of the peer in that session.

    In the current implementation, we don't offer representations of `target` so that parameter is ignored.
    Future releases will allow for this.
    """
    representation = await crud.get_working_representation(
        db, workspace_id, peer_id, options.session_id
    )
    return {"representation": representation}


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
    db=db,
):
    """Search a Peer"""
    embed_messages_enabled = settings.LLM.EMBED_MESSAGES
    if search.use_semantic_search and not embed_messages_enabled:
        raise DisabledException(
            "Semantic search requires EMBED_MESSAGES flag to be enabled"
        )

    stmt = await crud.search(
        search.query,
        workspace_name=workspace_id,
        peer_name=peer_id,
        use_semantic_search=search.use_semantic_search,
    )

    return await paginate(db, stmt)
