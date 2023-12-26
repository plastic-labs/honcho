from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .db import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/users/{user_id}/sessions", response_model=list[schemas.Session])
def get_sessions(user_id: str, db: Session = Depends(get_db)):
    return crud.get_sessions(db, user_id)


@app.post("/users/{user_id}/sessions", response_model=schemas.Session)
def create_session(
    user_id: str, session: schemas.SessionCreate, db: Session = Depends(get_db)
):
    return crud.create_session(db, user_id, session)


@app.get("/users/{user_id}/sessions/{session_id}", response_model=schemas.Session)
def get_session(user_id: str, session_id: int, db: Session = Depends(get_db)):
    db_session = crud.get_session(db, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return db_session


@app.post(
    "/users/{user_id}/sessions/{session_id}/messages/", response_model=schemas.Message
)
def create_message_for_session(
    user_id: str,
    session_id: int,
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
):
    return crud.create_message(db, message, session_id)


@app.get(
    "/users/{user_id}/sessions/{session_id}/messages/{message_id}/metacognitions/",
    response_model=list[schemas.Metacognitions],
)
def get_metacognitions_for_message(
    user_id: str,
    session_id: int,
    message_id: int,
    db: Session = Depends(get_db),
):
    return crud.get_metacognitions(db, message_id)


@app.post(
    "/users/{user_id}/sessions/{session_id}/messages/{message_id}/metacognitions/",
    response_model=schemas.Metacognitions,
)
def create_metacognition_for_message(
    user_id: str,
    session_id: int,
    message_id: int,
    metacognition: schemas.MetacognitionsCreate,
    db: Session = Depends(get_db),
):
    return crud.create_metacognition(db, metacognition, message_id)
