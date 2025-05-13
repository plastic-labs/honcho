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
    prefix="/apps/{app_id}/users/{user_id}/sessions",
    tags=["sessions"],
)


@router.get(
    "",
    response_model=schemas.Session,
)
async def get_session(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session_id: Optional[str] = Query(
        None, description="Session ID to retrieve. If not provided, uses JWT token"
    ),
    jwt_params: JWTParams = Depends(require_auth()),
    db=db,
):
    """
    Get a specific session for a user.

    If session_id is provided as a query parameter, it uses that (must match JWT session_id).
    Otherwise, it uses the session_id from the JWT token.
    """
    # Verify JWT has access to the requested resource
    if not jwt_params.ad:
        if jwt_params.ap is not None and jwt_params.ap != app_id:
            raise AuthenticationException("Unauthorized access to resource")
        if jwt_params.us is not None and jwt_params.us != user_id:
            raise AuthenticationException("Unauthorized access to resource")
    # If session_id provided in query, check if it matches jwt or user is admin
    if session_id:
        if (
            not jwt_params.ad
            and jwt_params.se is not None
            and jwt_params.se != session_id
        ):
            raise AuthenticationException("Unauthorized access to resource")
        target_session_id = session_id
    else:
        # Use session_id from JWT
        if not jwt_params.se:
            raise AuthenticationException(
                "Session ID not found in query parameter or JWT"
            )
        target_session_id = jwt_params.se

    # Let crud function handle the ResourceNotFoundException
    return await crud.get_session(
        db, app_id=app_id, session_id=target_session_id, user_id=user_id
    )


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def get_sessions(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    options: schemas.SessionGet = Body(
        ..., description="Filtering and pagination options for the sessions list"
    ),
    reverse: Optional[bool] = Query(
        False, description="Whether to reverse the order of results"
    ),
    db=db,
):
    """Get All Sessions for a User"""
    return await paginate(
        db,
        await crud.get_sessions(
            db,
            app_id=app_id,
            user_id=user_id,
            reverse=reverse,
            is_active=options.is_active,
            filter=options.filter,
        ),
    )


@router.post(
    "",
    response_model=schemas.Session,
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def create_session(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session: schemas.SessionCreate = Body(
        ..., description="Session creation parameters"
    ),
    db=db,
):
    """Create a Session for a User"""
    try:
        session_obj = await crud.create_session(
            db, app_id=app_id, user_id=user_id, session=session
        )
        logger.info(f"Session created successfully for user {user_id}")
        return session_obj
    except ValueError as e:
        logger.warning(f"Failed to create session: {str(e)}")
        raise ValidationException(str(e)) from e


@router.put(
    "/{session_id}",
    response_model=schemas.Session,
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def update_session(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session_id: str = Path(..., description="ID of the session to update"),
    session: schemas.SessionUpdate = Body(
        ..., description="Updated session parameters"
    ),
    db=db,
):
    """Update the metadata of a Session"""
    try:
        updated_session = await crud.update_session(
            db, app_id=app_id, user_id=user_id, session_id=session_id, session=session
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
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def delete_session(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session_id: str = Path(..., description="ID of the session to delete"),
    db=db,
):
    """Delete a session by marking it as inactive"""
    try:
        await crud.delete_session(
            db, app_id=app_id, user_id=user_id, session_id=session_id
        )
        logger.info(f"Session {session_id} deleted successfully")
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        logger.warning(f"Failed to delete session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e


@router.post(
    "/{session_id}/chat",
    response_model=schemas.DialecticResponse,
    responses={
        200: {
            "description": "Response to a question informed by Honcho's User Representation",
            "content": {"text/event-stream": {}},
        },
    },
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def chat(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session_id: str = Path(..., description="ID of the session"),
    options: schemas.DialecticOptions = Body(
        ..., description="Dialectic Endpoint Parameters"
    ),
    db=db,
):

    """Chat with the Dialectic API"""
    if not options.stream:
        return await agent.chat(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            queries=options.queries,
            db=db,
        )
    else:

        async def parse_stream():
            try:
                stream = await agent.chat(
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    queries=options.queries,
                    stream=True,
                    db=db,
                )
                if type(stream) is AsyncMessageStreamManager:
                    async with stream as stream_manager:
                        async for text in stream_manager.text_stream:
                            yield text
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        return StreamingResponse(
            content=parse_stream(), media_type="text/event-stream", status_code=200
        )


@router.get(
    "/{session_id}/clone",
    response_model=schemas.Session,
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def clone_session(
    app_id: str = Path(..., description="ID of the app"),
    user_id: str = Path(..., description="ID of the user"),
    session_id: str = Path(..., description="ID of the session to clone"),
    db=db,
    message_id: Optional[str] = Query(
        None, description="Message ID to cut off the clone at"
    ),
    deep_copy: bool = Query(False, description="Whether to deep copy metamessages"),
):
    """Clone a session, optionally up to a specific message"""
    try:
        cloned_session = await crud.clone_session(
            db,
            app_id=app_id,
            user_id=user_id,
            original_session_id=session_id,
            cutoff_message_id=message_id,
            deep_copy=deep_copy,
        )
        logger.info(f"Session {session_id} cloned successfully")
        return cloned_session
    except ValueError as e:
        logger.warning(f"Failed to clone session {session_id}: {str(e)}")
        raise ResourceNotFoundException("Session not found") from e
