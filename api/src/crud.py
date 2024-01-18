import json
from typing import Sequence, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas


def get_session(db: Session, session_id: int, user_id: Optional[str] = None):
    stmt = select(models.Session).where(models.Session.id == session_id)
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    return db.scalars(stmt).one_or_none()
    # return db.query(models.Session).filter(models.Session.id == session_id).first()


def get_sessions(db: Session, user_id: str, location_id: str | None = None) -> Sequence[schemas.Session]:
    stmt = select(models.Session).where(models.Session.user_id == user_id).where(models.Session.is_active.is_(True))

    if location_id is not None:
        stmt = stmt.where(models.Session.location_id == location_id)

    return db.scalars(stmt).all()

    # filtered_by_user = db.query(models.Session).filter(
    #     models.Session.user_id == user_id
    # )
    # filtered_by_location = (
    #     filtered_by_user.filter(models.Session.location_id == location_id)
    #     if location_id is not None
    #     else filtered_by_user
    # )
    # return (
    #     filtered_by_location.filter(models.Session.is_active.is_(True))
    #     .order_by(models.Session.created_at.desc())
    #     .all()
    # )

def create_session(db: Session, user_id: str, session: schemas.SessionCreate) -> models.Session:
    honcho_session = models.Session(
        user_id=user_id,
        location_id=session.location_id,
        session_data=json.dumps(session.session_data),
    )
    db.add(honcho_session)
    db.commit()
    db.refresh(honcho_session)
    return honcho_session

def update_session(db: Session, user_id: str, session_id: int, metadata: dict) -> bool:
    # stmt = select(models.Session).where(models.Session.id == session_id).where(models.Session.user_id == user_id)
    # honcho_session = db.scalars(stmt).one_or_none()
    honcho_session = get_session(db, session_id=session_id, user_id=user_id)
    # honcho_session = db.get(models.Session, session_id)
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")
    honcho_session.metadata = metadata
    db.commit()
    db.refresh(honcho_session)
    return honcho_session

def delete_session(db: Session, user_id: str, session_id: int) -> bool:
    stmt = select(models.Session).where(models.Session.id == session_id).where(models.Session.user_id == user_id)
    honcho_session = db.scalars(stmt).one_or_none()
    # honcho_session = db.get(models.Session, session_id)
    if honcho_session is None:
        return False
    honcho_session.is_active = False
    db.commit()
    return True

def create_message(db: Session, message: schemas.MessageCreate, user_id: str, session_id: int) -> models.Message:
    honcho_session = get_session(db, session_id, user_id)
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_message = models.Message(
        session_id=session_id,
        is_user=message.is_user,
        content=message.content,
    )
    db.add(honcho_message)
    db.commit()
    db.refresh(honcho_message)
    return honcho_message

def get_messages(db: Session, user_id: str, session_id: int) -> Sequence[schemas.Message]:
    session = get_session(db, session_id, user_id)
    if session is None:
        raise ValueError("Session not found or does not belong to user")
    stmt = select(models.Message).where(models.Message.session_id == session_id)
    return db.scalars(stmt).all()
    # return (
    #     db.query(models.Message)
    #     .filter(models.Message.session_id == session_id)
    #     .all()
    # )

# def get_metacognitions(db: Session, message_id: int):
#     return (
#         db.query(models.Metacognitions)
#         .filter(models.Metacognitions.message_id == message_id)
#         .all()
#     )

# def create_metacognition(
#     db: Session, metacognition: schemas.MetacognitionsCreate, message_id: int
# ):
#     honcho_metacognition = models.Metacognitions(
#         message_id=message_id,
#         metacognition_type=metacognition.metacognition_type,
#         content=metacognition.content,
#     )
#     db.add(honcho_metacognition)
#     db.commit()
#     db.refresh(honcho_metacognition)
#     return honcho_metacognition
