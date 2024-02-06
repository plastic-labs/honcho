from fastapi import Depends, FastAPI, HTTPException, APIRouter, Request
from typing import Optional
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi_pagination import Page, add_pagination
from fastapi_pagination.ext.sqlalchemy import paginate
# import uvicorn

from . import crud, models, schemas
from .db import SessionLocal, engine

models.Base.metadata.create_all(bind=engine) # Scaffold Database if not already done

app = FastAPI()

router = APIRouter(prefix="/apps/{app_id}/users/{user_id}")

# Create a Limiter instance
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# Add SlowAPI middleware to the application
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


add_pagination(app)

def get_db():
    """FastAPI Dependency Generator for Database"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

########################################################
# Session Routes
########################################################

@router.get("/sessions", response_model=Page[schemas.Session])
def get_sessions(request: Request, app_id: str, user_id: str, location_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Get All Sessions for a User

    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        location_id (str, optional): Optional Location ID representing the location of a session

    Returns:
        list[schemas.Session]: List of Session objects 

    """
    # if location_id is not None:
        # return paginate(db, crud.get_sessions(db, app_id=app_id, user_id=user_id, location_id=location_id))
    #     return crud.get_sessions(db, app_id=app_id, user_id=user_id, location_id=location_id)
    return paginate(db, crud.get_sessions(db, app_id=app_id, user_id=user_id, location_id=location_id))
    # return crud.get_sessions(db, app_id=app_id, user_id=user_id)


@router.post("/sessions", response_model=schemas.Session)
def create_session(
        request: Request, app_id: str, user_id: str, session: schemas.SessionCreate, db: Session = Depends(get_db)
):
    """Create a Session for a User
        
    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session (schemas.SessionCreate): The Session object containing any metadata and a location ID

    Returns:
        schemas.Session: The Session object of the new Session
        
    """
    return crud.create_session(db, app_id=app_id, user_id=user_id, session=session)

@router.put("/sessions/{session_id}", response_model=schemas.Session)
def update_session(
    request: Request, 
    app_id: str,
    user_id: str,
    session_id: int,
    session: schemas.SessionUpdate,
    db: Session = Depends(get_db),
    ):
    """Update the metadata of a Session
    
    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to update
        session (schemas.SessionUpdate): The Session object containing any new metadata

    Returns:
        schemas.Session: The Session object of the updated Session

    """
    if session.session_data is None:
        raise HTTPException(status_code=400, detail="Session data cannot be empty") # TODO TEST if I can set the metadata to be blank with this 
    try:
        return crud.update_session(db, app_id=app_id, user_id=user_id, session_id=session_id, session_data=session.session_data) 
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

@router.delete("/sessions/{session_id}")
def delete_session(
    request: Request, 
    app_id: str,
    user_id: str,
    session_id: int,
    db: Session = Depends(get_db),
    ):
    """Delete a session by marking it as inactive

    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to delete

    Returns:
        dict: A message indicating that the session was deleted

    Raises:
        HTTPException: If the session is not found

    """
    response = crud.delete_session(db, app_id=app_id, user_id=user_id, session_id=session_id)
    if response:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/sessions/{session_id}", response_model=schemas.Session)
def get_session(request: Request, app_id: str, user_id: str, session_id: int, db: Session = Depends(get_db)):
    """Get a specific session for a user by ID

    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve

    Returns: 
        schemas.Session: The Session object of the requested Session

    Raises:
        HTTPException: If the session is not found
    """
    honcho_session = crud.get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    if honcho_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_session

########################################################
# Message Routes
########################################################

@router.post(
    "/sessions/{session_id}/messages",
    response_model=schemas.Message
)
def create_message_for_session(
    request: Request, 
    app_id: str,
    user_id: str,
    session_id: int,
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    """Adds a message to a session

    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to add the message to
        message (schemas.MessageCreate): The Message object to add containing the message content and type

    Returns:
        schemas.Message: The Message object of the added message

    Raises:
        HTTPException: If the session is not found

    """
    try:
        return crud.create_message(db, message=message, app_id=app_id, user_id=user_id, session_id=session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get(
    "/sessions/{session_id}/messages", 
    response_model=Page[schemas.Message]
)
def get_messages_for_session(
    request: Request, 
    app_id: str,
    user_id: str,
    session_id: int,
    db: Session = Depends(get_db),
):
    """Get all messages for a session
    
    Args:
        app_id (str): The ID of the app representing the client application using honcho
        user_id (str): The User ID representing the user, managed by the user
        session_id (int): The ID of the Session to retrieve

    Returns:
        list[schemas.Message]: List of Message objects

    Raises:
        HTTPException: If the session is not found

    """
    try: 
        return paginate(db, crud.get_messages(db, app_id=app_id, user_id=user_id, session_id=session_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")


app.include_router(router)

########################################################
# Metacognition Routes
########################################################

# @app.get(
#     "/users/{user_id}/sessions/{session_id}/messages/{message_id}/metacognitions/",
#     response_model=list[schemas.Metacognitions],
# )
# def get_metacognitions_for_message(
#     user_id: str,
#     session_id: int,
#     message_id: int,
#     db: Session = Depends(get_db),
# ):
#     return crud.get_metacognitions(db, message_id)


# @app.post(
#     "/users/{user_id}/sessions/{session_id}/messages/{message_id}/metacognitions/",
#     response_model=schemas.Metacognitions,
# )
# def create_metacognition_for_message(
#     user_id: str,
#     session_id: int,
#     message_id: int,
#     metacognition: schemas.MetacognitionsCreate,
#     db: Session = Depends(get_db),
# ):
#     return crud.create_metacognition(db, metacognition, message_id)

