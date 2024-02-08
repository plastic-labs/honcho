from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Uuid
import uuid
import datetime
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .db import Base


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
    app_id: Mapped[str] = mapped_column(String(512), index=True)
    user_id: Mapped[str] = mapped_column(String(512), index=True)
    location_id: Mapped[str] = mapped_column(String(512), index=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    session_data: Mapped[str] 
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)
    messages = relationship("Message", back_populates="session")

    def __repr__(self) -> str:
        return f"Session(id={self.id}, app_id={self.app_id}, user_id={self.user_id}, location_id={self.location_id}, is_active={self.is_active}, created_at={self.created_at})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"))
    is_user: Mapped[bool]
    content: Mapped[str]  = mapped_column(String(65535)) 

    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)
    session = relationship("Session", back_populates="messages")
    metamessages = relationship("Metamessage", back_populates="message")
    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, is_user={self.is_user}, content={self.content[10:]})"


class Metamessage(Base):
    __tablename__ = "metamessages"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, index=True, default=uuid.uuid4)
    metamessage_type: Mapped[str] = mapped_column(String(512), index=True) 
    content: Mapped[str] = mapped_column(String(65535)) 
    message_id = Column(Uuid, ForeignKey("messages.id"))

    message = relationship("Message", back_populates="metamessages")
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)

    def __repr__(self) -> str:
        return f"Metamessages(id={self.id}, message_id={self.message_id}, metamessage_type={self.metamessage_type}, content={self.content[10:]})"
