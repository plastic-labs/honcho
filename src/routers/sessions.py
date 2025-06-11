import logging
from typing import Optional

from anthropic import AsyncMessageStreamManager
from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import agent, crud, schemas
from src.dependencies import db
from src.exceptions import (
    AuthenticationException,
    ResourceNotFoundException,
    ValidationException,
)
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
    if not jwt_params.ad:
        if jwt_params.ap is not None and jwt_params.ap != workspace_id:
            raise AuthenticationException("Unauthorized access to resource")

    # Use peer_id from JWT if not provided in query
    if not session.name:
        if not jwt_params.se:
            raise AuthenticationException("Session ID not found in query parameter or JWT")
        session.name = jwt_params.se

    # Let crud function handle the ResourceNotFoundException
    return await crud.get_or_create_session(
        db, workspace_name=workspace_id, session=session
    )


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(app_id="workspace_id"))],
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
        if hasattr(options, 'filter') and options.filter:
            filter_param = options.filter
            if filter_param == {}: # Explicitly check for empty dict
                filter_param = None
        if hasattr(options, 'is_active'): # Check if is_active is present
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
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def update_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session to update"),
    session: schemas.SessionUpdate = Body(
        ..., description="Updated session parameters"
    ),
    peer_id: Optional[str] = Query(
        None, description="Peer ID to verify access"
    ),
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
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
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
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
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
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def add_peers_to_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: list[str] = Body(..., description="List of peer IDs to add to the session"),
    db=db,
):
    """Add peers to a session"""
    pass
    # try:
    #     # TODO: Implement crud.add_peers_to_session
    #     updated_session = await crud.add_peers_to_session(
    #         db, workspace_name=workspace_id, session_name=session_id, peers=peers
    #     )
    #     logger.info(f"Added peers to session {session_id} successfully")
    #     return updated_session
    # except ValueError as e:
    #     logger.warning(f"Failed to add peers to session {session_id}: {str(e)}")
    #     raise ResourceNotFoundException("Session not found") from e


@router.put(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def set_session_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: list[str] = Body(..., description="List of peer IDs to set for the session"),
    db=db,
):
    """Set the peers in a session"""
    pass
    # try:
    #     # TODO: Implement crud.set_session_peers
    #     updated_session = await crud.set_session_peers(
    #         db, workspace_name=workspace_id, session_name=session_id, peers=peers
    #     )
    #     logger.info(f"Set peers for session {session_id} successfully")
    #     return updated_session
    # except ValueError as e:
    #     logger.warning(f"Failed to set peers for session {session_id}: {str(e)}")
    #     raise ResourceNotFoundException("Session not found") from e


@router.delete(
    "/{session_id}/peers",
    response_model=schemas.Session,
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def remove_peers_from_session(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    peers: list[str] = Body(..., description="List of peer IDs to remove from the session"),
    db=db,
):
    """Remove peers from a session"""
    pass
    # try:
    #     # TODO: Implement crud.remove_peers_from_session
    #     updated_session = await crud.remove_peers_from_session(
    #         db, workspace_name=workspace_id, session_name=session_id, peers=peers
    #     )
    #     logger.info(f"Removed peers from session {session_id} successfully")
    #     return updated_session
    # except ValueError as e:
    #     logger.warning(f"Failed to remove peers from session {session_id}: {str(e)}")
    #     raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/peers",
    response_model=list[str],
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def get_session_peers(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    db=db,
):
    """Get peers from a session"""
    pass
    # try:
    #     # TODO: Implement crud.get_session_peers
    #     peers = await crud.get_session_peers(
    #         db, workspace_name=workspace_id, session_name=session_id
    #     )
    #     return peers
    # except ValueError as e:
    #     logger.warning(f"Failed to get peers from session {session_id}: {str(e)}")
    #     raise ResourceNotFoundException("Session not found") from e


@router.get(
    "/{session_id}/context",
    response_model=schemas.SessionContext,
    dependencies=[
        Depends(
            require_auth(app_id="workspace_id", session_id="session_id")
        )
    ],
)
async def get_session_context(
    workspace_id: str = Path(..., description="ID of the workspace"),
    session_id: str = Path(..., description="ID of the session"),
    db=db,
):
    pass
    # """Get context from a session"""
    # try:
    #     # TODO: Implement crud.get_session_context
    #     context = await crud.get_session_context(
    #         db, workspace_name=workspace_id, session_name=session_id
    #     )
    #     return context
    # except ValueError as e:
    #     logger.warning(f"Failed to get context from session {session_id}: {str(e)}")
    #     raise ResourceNotFoundException("Session not found") from e

