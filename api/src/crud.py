import json
import uuid
from typing import Optional

from sqlalchemy import select, Select
from sqlalchemy.orm import Session

from . import models, schemas


def get_session(db: Session, app_id: str, session_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[models.Session]:
    stmt = select(models.Session).where(models.Session.app_id == app_id).where(models.Session.id == session_id)
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    session = db.scalars(stmt).one_or_none()
    return session
    # return db.query(models.Session).filter(models.Session.id == session_id).first()


def get_sessions(
        db: Session, app_id: str, user_id: str, location_id: str | None = None
) -> Select:
    stmt = (
        select(models.Session)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Session.is_active.is_(True))
        .order_by(models.Session.created_at)
    )

    if location_id is not None:
        stmt = stmt.where(models.Session.location_id == location_id)

    return stmt
    # return db.scalars(stmt).all()

def create_session(
    db: Session, app_id: str, user_id: str, session: schemas.SessionCreate
) -> models.Session:
    honcho_session = models.Session(
        app_id=app_id,
        user_id=user_id,
        location_id=session.location_id,
        session_data=json.dumps(session.session_data),
    )
    db.add(honcho_session)
    db.commit()
    db.refresh(honcho_session)
    return honcho_session


def update_session(db: Session, app_id: str, user_id: str, session_id: uuid.UUID, session_data: dict) -> bool:
    honcho_session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")
    honcho_session.session_data = json.dumps(session_data)
    db.commit()
    db.refresh(honcho_session)
    return honcho_session


def delete_session(db: Session, app_id: str, user_id: str, session_id: uuid.UUID) -> bool:
    stmt = (
        select(models.Session)
        .where(models.Session.id == session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )
    honcho_session = db.scalars(stmt).one_or_none()
    if honcho_session is None:
        return False
    honcho_session.is_active = False
    db.commit()
    return True


def create_message(
        db: Session, message: schemas.MessageCreate, app_id: str, user_id: str, session_id: uuid.UUID
) -> models.Message:
    honcho_session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
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


def get_messages(
    db: Session, app_id: str, user_id: str, session_id: uuid.UUID
) -> Select:
    # session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    # if session is None:
    #     raise ValueError("Session not found or does not belong to user")
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.id == models.Message.session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at)
    )
    return stmt
    # return db.scalars(stmt).all()
    # return (
    #     db.query(models.Message)
    #     .filter(models.Message.session_id == session_id)
    #     .all()
    # )

def get_message(
        db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: uuid.UUID     
) -> Optional[models.Message]:
    # session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    # if session is None:
    #     raise ValueError("Session not found or does not belong to user")
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.id == models.Message.session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Message.id == message_id)

    )
    return db.scalars(stmt).one_or_none()


def get_metamessages(db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: Optional[uuid.UUID], metamessage_type: Optional[str] = None) -> Select:
    stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .order_by(models.Metamessage.created_at)
    )
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)
    if metamessage_type is not None:
        stmt = stmt.where(models.Metamessage.metamessage_type == metamessage_type)
    return stmt

def get_metamessage(
        db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: uuid.UUID, metamessage_id: uuid.UUID
) -> Optional[models.Metamessage]: 
    stmt = (
         select(models.Metamessage)
        .join(models.Message, models.Message.id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Metamessage.message_id == message_id)
        .where(models.Metamessage.id == metamessage_id)
       
    )
    return db.scalars(stmt).one_or_none()

def create_metamessage(
    db: Session,
    metamessage: schemas.MetamessageCreate,
    app_id: str,
    user_id: str,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
):
    message = get_message(db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id)
    if message is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_metamessage = models.Metamessage(
        message_id=message_id,
        metamessage_type=metamessage.metamessage_type,
        content=metamessage.content,
    )

    db.add(honcho_metamessage)
    db.commit()
    db.refresh(honcho_metamessage)
    return honcho_metamessage
