from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
import datetime
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .db import Base


class Session(Base):
    __tablename__ = "sessions"
    # id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # user_id = Column(String, index=True)
    # location_id = Column(String, index=True)
    # is_active = Column(Boolean, default=True)
    # session_data = Column(String)
    # created_at = Column(DateTime, default=datetime.datetime.utcnow)
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(index=True)
    location_id: Mapped[str] = mapped_column(index=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    session_data: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)
    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"
    # id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # session_id = Column(Integer, ForeignKey("sessions.id"))
    # is_user = Column(Boolean)
    # content = Column(String)
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    is_user: Mapped[bool]
    content: Mapped[str] # TODO add a max message length

    session = relationship("Session", back_populates="messages")
    metacognitions = relationship("Metacognitions", back_populates="message")


# TODO: add metacognitive data to messages
class Metacognitions(Base):
    __tablename__ = "metacognitions"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    metacognition_type: Mapped[str] # TODO add a max metacognition type length
    content: Mapped[str]
    # id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    # metacognition_type = Column(String, index=True)
    # content = Column(String)

    message = relationship("Message", back_populates="metacognitions")
