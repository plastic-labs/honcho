from typing import Optional

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
    """Get All Sessions for a User

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user

    Returns:
        list[schemas.Session]: List of Session objects

    """
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
    """Create a Session for a User

    Args:
        app_id (str): The ID of the app representing the client
        application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session (schemas.SessionCreate): The Session object containing any
        metadata

    Returns:
        schemas.Session: The Session object of the new Session

    """
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
    """Update the metadata of a Session

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (str): The ID of the Session to update
        session (schemas.SessionUpdate): The Session object containing any new metadata

    Returns:
        schemas.Session: The Session object of the updated Session

    """
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
    """Delete a session by marking it as inactive

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (str): The ID of the Session to delete

    Returns:
        dict: A message indicating that the session was deleted

    Raises:
        HTTPException: If the session is not found

    """
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
    """Get a specific session for a user by ID

    Args:
        app_id (str): The ID of the app representing the client application using
        honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (str): The ID of the Session to retrieve

    Returns:
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
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
    print(query)
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
    async def parse_stream():
        stream = await agent.chat(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            query=query,
            stream=True,
        )
        for chunk in stream:
            yield chunk.content

    return StreamingResponse(
        content=parse_stream(), media_type="text/event-stream", status_code=200
    )


@router.get("/{session_id}/clone", response_model=schemas.Session)
async def clone_session(
    app_id: str,
    user_id: str,
    session_id: str,
    db=db,
    messageId: Optional[str] = None,
    deepCopy: bool = False,
):
    return await crud.clone_session(
        db,
        app_id=app_id,
        user_id=user_id,
        original_session_id=session_id,
        cutoff_message_id=messageId,
        deep_copy=deepCopy,
    )
