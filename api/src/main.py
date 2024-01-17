from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
import uvicorn

from . import crud, models, schemas
from .db import SessionLocal, engine

models.Base.metadata.create_all(bind=engine) # Scaffold Database if not already done

app = FastAPI()

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

@app.get("/users/{user_id}/sessions", response_model=list[schemas.Session])
def get_sessions(user_id: str, db: Session = Depends(get_db)):
    return crud.get_sessions(db, user_id)


@app.post("/users/{user_id}/sessions", response_model=schemas.Session)
def create_session(
    user_id: str, session: schemas.SessionCreate, db: Session = Depends(get_db)
):
    return crud.create_session(db, user_id, session)

@app.put("/users/{user_id}/sessions/{session_id}", response_model=schemas.Session)
def update_session(
    user_id: str,
    session_id: int,
    session: schemas.SessionUpdate,
    db: Session = Depends(get_db),
    ):
    if session.session_data is None:
        raise HTTPException(status_code=400, detail="Session data cannot be empty") # TODO TEST if I can set the metadata to be blank with this 
    try:
        return crud.update_session(db, user_id=user_id, session_id=session_id, metadata=session.session_data) 
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/users/{user_id}/sessions/{session_id}")
def delete_session(
    user_id: str,
    session_id: int,
    db: Session = Depends(get_db),
    ):
    response = crud.delete_session(db, user_id=user_id, session_id=session_id)
    if response:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/users/{user_id}/sessions/{session_id}", response_model=schemas.Session)
def get_session(user_id: str, session_id: int, db: Session = Depends(get_db)):
    honcho_session = crud.get_session(db, session_id=session_id, user_id=user_id)
    if honcho_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return honcho_session

########################################################
# Message Routes
########################################################

@app.post(
    "/users/{user_id}/sessions/{session_id}/messages",
    response_model=schemas.Message
)
def create_message_for_session(
    user_id: str,
    session_id: int,
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    try:
        return crud.create_message(db, message=message, user_id=user_id, session_id=session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get(
    "/users/{user_id}/sessions/{session_id}/messages", 
    response_model=list[schemas.Message]
)
def get_messages_for_session(
        user_id: str,
        session_id: int,
        db: Session = Depends(get_db),
        ):
    try: 
        return crud.get_messages(db, user_id=user_id, session_id=session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

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

