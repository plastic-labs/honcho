from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
import datetime
from sqlalchemy.orm import relationship

from .db import Base


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True)
    location_id = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    session_data = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    message_type = Column(String, index=True)
    content = Column(String)

    session = relationship("Session", back_populates="messages")


# TODO: add metacognitive data to messages
