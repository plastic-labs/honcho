import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import crud, schemas
from src.dependencies import db
from src.exceptions import AuthenticationException, ResourceNotFoundException
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/sessions",
    tags=["sessions"],
)


@router.post(
    "",
    response_model=schemas.Session,
)
async def get_or_create_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session: schemas.SessionCreate = Body(
        ..., description="Session creation parameters"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a specific session in a workspace.

    If peer_id is provided as a query parameter, it verifies the peer is in the session.
    Otherwise, it uses the peer_id from the JWT token for verification.
    """
    # Verify JWT has access to the requested resource
    if not jwt_params.ad and jwt_params.w is not None and jwt_params.w != workspace_id:
        raise AuthenticationException("Unauthorized access to resource")

    # Use peer_id from JWT if not provided in query
    if session.name:
        if (
            not jwt_params.ad
            and jwt_params.s is not None
            and jwt_params.s != session.name
        ):
            raise AuthenticationException("Unauthorized access to resource")
    else:
        if not jwt_params.s:
            raise AuthenticationException(
                "Session ID not found in query parameter or JWT"
            )
        session.name = jwt_params.s

    # Let crud function handle the ResourceNotFoundException
    return await crud.get_or_create_session(
        db, workspace_name=workspace_id, session=session
    )


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)
async def get_sessions(
    workspace_id: str = Path(..., description="ID of the workspace"),
    options: Optional[schemas.SessionGet] = Body(
        None, description="Filtering and pagination options for the sessions list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    db=db,
):
    """Get All Sessions in a Workspace"""
    filter_param = None
    is_active_param = False  # Default from schema

    if options:
        if hasattr(options, "filter") and options.filter:
            filter_param = options.filter
            if filter_param == {}:  # Explicitly check for empty dict
                filter_param = None
        if hasattr(options, "is_active"):  # Check if is_active is present
            is_active_param = options.is_active

    return await paginate(
        db,
        await crud.get_sessions(
            workspace_name=workspace_id,
            reverse=reverse,
            is_active=is_active_param,
            filter=filter_param,
        ),
    )


@router.put(
    "/{session_id}",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def update_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to update"),
    session: schemas.SessionUpdate = Body(
        ..., description="Updated session parameters"
    ),
    peer_id: Optional[str] = Query(None, description="Peer ID to verify access"),
    db=db,
):
    """Update the metadata of a Session"""
    try:
        updated_session = await crud.update_session(
            db, workspace_name=workspace_id, session_name=session_id, session=session
        )
        logger.info(f"Session {session_id} updated successfully")
        return updated_session
    except ValueError as e:
        logger.warning(f"Failed to update session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.delete(
    "/{session_id}",
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def delete_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to delete"),
    db=db,
):
    """Delete a session by marking it as inactive"""
    try:
        await crud.delete_session(
            db, workspace_name=workspace_id, session_name=session_id
        )
        logger.info(f"Session {session_id} deleted successfully")
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        logger.warning(f"Failed to delete session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


# TODO: Update chat endpoint to work with new workspace/peer paradigm
# This endpoint needs significant rework for multi-peer sessions
# @router.post(
#     "/{session_id}/chat",
#     response_model=schemas.DialecticResponse,
#     responses={
#         200: {
#             "description": "Response to a question informed by Honcho's User Representation",
#             "content": {"text/event-stream": {}},
#         },
#     },
#     dependencies=[
#         Depends(
#             require_auth(app_id="workspace_id", session_id="session_id")
#         )
#     ],
# )
# async def chat(
#     workspace_id: str = Path(..., description="ID of the workspace"),
#     session_id: str = Path(..., description="ID of the session"),
#     peer_id: str = Query(..., description="ID of the peer making the request"),
#     options: schemas.DialecticOptions = Body(
#         ..., description="Dialectic Endpoint Parameters"
#     ),
# ):

#     """Chat with the Dialectic API"""
#     # TODO: Update agent.chat to work with workspace_id/peer_id instead of app_id/user_id
#     if not options.stream:
#         return await agent.chat(
#             app_id=workspace_id,  # Temporary mapping
#             user_id=peer_id,      # Temporary mapping
#             session_id=session_id,
#             queries=options.queries,
#         )
#     else:

#         async def parse_stream():
#             try:
#                 stream = await agent.chat(
#                     app_id=workspace_id,  # Temporary mapping
#                     user_id=peer_id,      # Temporary mapping
#                     session_id=session_id,
#                     queries=options.queries,
#                     stream=True,
#                 )
#                 if type(stream) is AsyncMessageStreamManager:
#                     async with stream as stream_manager:
#                         async for text in stream_manager.text_stream:
#                             yield text
#             except Exception as e:
#                 logger.error(f"Error in stream: {str(e)}")
#                 raise HTTPException(status_code=500, detail=str(e)) from e

#         return StreamingResponse(
#             content=parse_stream(), media_type="text/event-stream", status_code=200
#         )


@router.get(
    "/{session_id}/clone",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def clone_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to clone"),
    db=db,
    message_id: Optional[str] = Query(
        None, description="Message ID to cut off the clone at"
    ),
):
    """Clone a session, optionally up to a specific message"""
    try:
        # TODO: Update crud.clone_session to work with new paradigm
        cloned_session = await crud.clone_session(
            db,
            workspace_name=workspace_id,
            original_session_name=session_id,
            cutoff_message_id=message_id,
        )
        logger.info(f"Session {session_id} cloned successfully")
        return cloned_session
    except ValueError as e:
        logger.warning(f"Failed to clone session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def add_peers_to_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ..., description="List of peer IDs to add to the session"
    ),
    db=db,
):
    """Add peers to a session"""
    try:
        workspace_name, session_name = workspace_id, session_id
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(
                name=session_name,
                peer_names=peers,
            ),
            workspace_name=workspace_name,
        )
        logger.info(f"Added peers to session {session_name} successfully")
        return session
    except ValueError as e:
        logger.warning(f"Failed to add peers to session {session_name}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.put(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def set_session_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: dict[str, schemas.SessionPeerConfig] = Body(
        ..., description="List of peer IDs to set for the session"
    ),
    db=db,
):
    """Set the peers in a session"""
    try:
        workspace_name, session_name = workspace_id, session_id
        await crud.set_peers_for_session(
            db,
            workspace_name=workspace_name,
            session_name=session_name,
            peer_names=peers,
        )
        # Get the session to return
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_name),
            workspace_name=workspace_name,
        )
        logger.info(f"Set peers for session {session_name} successfully")
        return session
    except ValueError as e:
        logger.warning(f"Failed to set peers for session {session_name}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.delete(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def remove_peers_from_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: list[str] = Body(
        ..., description="List of peer IDs to remove from the session"
    ),
    db=db,
):
    """Remove peers from a session"""
    try:
        session_name = session_id
        await crud.remove_peers_from_session(
            db,
            workspace_name=workspace_id,
            session_name=session_name,
            peer_names=set(peers),
        )
        # Get the session to return
        session = await crud.get_or_create_session(
            db,
            session=schemas.SessionCreate(name=session_name),
            workspace_name=workspace_id,
        )
        logger.info(f"Removed peers from session {session_name} successfully")
        return session
    except ValueError as e:
        logger.warning(f"Failed to remove peers from session {session_name}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/peers/{peer_id}/config",
    response_model=schemas.SessionPeerConfig,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_peer_config(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peer_id: str = Path(..., description="ID of the peer"),
    db=db,
):
    """Get the configuration for a peer in a session"""
    return await crud.get_peer_config(
        db,
        workspace_name=workspace_id,
        session_name=session_id,
        peer_id=peer_id,
    )


@router.post(
    "/{session_id}/peers/{peer_id}/config",
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def set_peer_config(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peer_id: str = Path(..., description="ID of the peer"),
    config: schemas.SessionPeerConfig = Body(..., description="Peer configuration"),
    db=db,
):
    """Set the configuration for a peer in a session"""
    try:
        await crud.set_peer_config(
            db,
            workspace_name=workspace_id,
            session_name=session_id,
            peer_id=peer_id,
            config=config,
        )
        logger.info(
            f"Set peer config for {peer_id} in session {session_id} successfully"
        )
        return Response(status_code=200)
    except ValueError as e:
        logger.warning(
            f"Failed to set peer config for {peer_id} in session {session_id}: {str(e)}"
        )
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/peers",
    response_model=Page[schemas.Peer],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_session_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    db=db,
):
    """Get peers from a session"""
    try:
        session_name = session_id
        peers_query = await crud.get_peers_from_session(
            workspace_name=workspace_id, session_name=session_name
        )
        return await paginate(db, peers_query)
    except ValueError as e:
        logger.warning(f"Failed to get peers from session {session_name}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/context",
    response_model=schemas.SessionContext,
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def get_session_context(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    tokens: Optional[int] = Query(
        None,
        description="Number of tokens to use for the context. Includes summary if set to true",
    ),
    summary: bool = Query(
        False,
        description="Whether to summarize the session history prior to the cutoff message",
    ),  # default to false
    db=db,
):
    messages_stmt = await crud.get_messages(
        workspace_name=workspace_id,
        session_name=session_id,
        token_limit=tokens,
    )
    result = await db.execute(messages_stmt)
    messages = list(result.scalars().all())

    return schemas.SessionContext(
        name=session_id,
        messages=messages,
        summary="TODO: give a summary" if summary else "",
    )


@router.post(
    "/{session_id}/search",
    response_model=Page[schemas.Message],
    dependencies=[
        Depends(require_auth(workspace_name="workspace_id", session_name="session_id"))
    ],
)
async def search_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    query: str = Body(..., description="Search query"),
    db=db,
):
    """Search a Session"""
    stmt = await crud.search(query, workspace_id=workspace_id, session_id=session_id)

    return await paginate(db, stmt)
