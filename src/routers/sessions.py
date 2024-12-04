from typing import Optional

from anthropic import MessageStreamManager
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate

from src import agent, crud, schemas
from src.dependencies import db
from src.security import auth

router = APIRouter(
    prefix="/apps/{app_id}/users/{user_id}/sessions",
    tags=["sessions"],
    dependencies=[Depends(auth)],
)


@router.post("/list", response_model=Page[schemas.Session])
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


@router.post("", response_model=schemas.Session)
async def create_session(
    app_id: str,
    user_id: str,
    session: schemas.SessionCreate,
    db=db,
):
    """Create a Session for a User"""
    try:
        value = await crud.create_session(
            db, app_id=app_id, user_id=user_id, session=session
        )
        return value
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # except Exception as e:
    #     print(e)
    #     raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/{session_id}", response_model=schemas.Session)
async def update_session(
    app_id: str,
    user_id: str,
    session_id: str,
    session: schemas.SessionUpdate,
    db=db,
):
    """Update the metadata of a Session"""
    if session.metadata is None:
        raise HTTPException(status_code=400, detail="Session metadata cannot be empty")
    try:
        return await crud.update_session(
            db, app_id=app_id, user_id=user_id, session_id=session_id, session=session
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None


@router.delete("/{session_id}")
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
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Session not found") from e


@router.get("/{session_id}", response_model=schemas.Session)
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
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_session


@router.post("/{session_id}/chat", response_model=schemas.AgentChat)
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


@router.get("/{session_id}/clone", response_model=schemas.Session)
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
