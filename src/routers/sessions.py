import logging
from typing import Optional

from anthropic import MessageStreamManager
from fastapi import APIRouter, Depends
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
from src.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions",
    tags=["sessions"],
)

jwt_params = Depends(require_auth(session_id="session_id"))


@router.get(
    "",
    response_model=schemas.Session,
)
async def get_session_from_token(
    app_id: str,
    user_id: str,
    jwt_params=jwt_params,
    db=db,
):
    """
    Get a specific session for a user by session_id provided in the JWT.
    If no session_id is provided, return a 401 Unauthorized error.
    """
    if jwt_params.se is None:
        raise AuthenticationException("Session not found in JWT")
    return await crud.get_session(
        db, app_id=app_id, session_id=jwt_params.se, user_id=user_id
    )


@router.post(
    "/list",
    response_model=Page[schemas.Session],
    dependencies=[Depends(require_auth(app_id="app_id", user_id="user_id"))],
)
async def get_sessions(
    app_id: str,
    user_id: str,
    options: schemas.SessionGet,
    reverse: Optional[bool] = False,
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
    app_id: str,
    user_id: str,
    session: schemas.SessionCreate,
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
    app_id: str,
    user_id: str,
    session_id: str,
    session: schemas.SessionUpdate,
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
    app_id: str,
    user_id: str,
    session_id: str,
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


@router.get(
    "/{session_id}",
    response_model=schemas.Session,
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def get_session(
    app_id: str,
    user_id: str,
    session_id: str,
    db=db,
):
    """Get a specific session for a user by ID"""
    honcho_session = await crud.get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        logger.warning(f"Session {session_id} not found for user {user_id}")
        raise ResourceNotFoundException(f"Session with ID {session_id} not found")
    return honcho_session


@router.post(
    "/{session_id}/chat",
    response_model=schemas.AgentChat,
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def chat(
    app_id: str,
    user_id: str,
    session_id: str,
    query: schemas.AgentQuery,
):
    """Chat with the Dialectic API"""
    return await agent.chat(
        app_id=app_id, user_id=user_id, session_id=session_id, query=query
    )


@router.post(
    "/{session_id}/chat/stream",
    responses={
        200: {
            "description": "Chat stream",
            "content": {
                "text/event-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
    dependencies=[
        Depends(
            require_auth(app_id="app_id", user_id="user_id", session_id="session_id")
        )
    ],
)
async def get_chat_stream(
    app_id: str,
    user_id: str,
    session_id: str,
    query: schemas.AgentQuery,
):
    """Stream Results from the Dialectic API"""

    async def parse_stream():
        stream = await agent.chat(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            query=query,
            stream=True,
        )
        if type(stream) is MessageStreamManager:
            with stream as stream_manager:
                for text in stream_manager.text_stream:
                    yield text

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
    app_id: str,
    user_id: str,
    session_id: str,
    db=db,
    message_id: Optional[str] = None,
    deep_copy: bool = False,
):
    """Clone a session for a user, optionally will deep clone metamessages as well"""
    return await crud.clone_session(
        db,
        app_id=app_id,
        user_id=user_id,
        original_session_id=session_id,
        cutoff_message_id=message_id,
        deep_copy=deep_copy,
    )
