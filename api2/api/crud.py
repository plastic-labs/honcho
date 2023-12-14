from sqlalchemy.orm import Session

from . import models, schemas

import json


def get_session(db: Session, session_id: int):
    return db.query(models.Session).filter(models.Session.id == session_id).first()


def get_sessions(db: Session, user_id: str, location_id: str | None = None):
    filtered_by_user = db.query(models.Session).filter(
        models.Session.user_id == user_id
    )
    filtered_by_location = (
        filtered_by_user.filter(models.Session.location_id == location_id)
        if location_id is not None
        else filtered_by_user
    )
    return (
        filtered_by_location.filter(models.Session.is_active == True)
        .order_by(models.Session.created_at.desc())
        .all()
    )


def create_session(db: Session, user_id: str, session: schemas.SessionCreate):
    db_session = models.Session(
        user_id=user_id,
        location_id=session.location_id,
        session_data=json.dumps(session.session_data),
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def create_message(db: Session, message: schemas.MessageCreate, session_id: int):
    db_message = models.Message(
        session_id=session_id,
        message_type=message.message_type,
        content=message.content,
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message
