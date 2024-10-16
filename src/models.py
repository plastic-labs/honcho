import datetime

from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Identity,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base

load_dotenv()


class App(Base):
    __tablename__ = "apps"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(String(512), index=True, unique=True)
    users = relationship("User", back_populates="app")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(String(512), index=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    app = relationship("App", back_populates="users")
    sessions = relationship("Session", back_populates="user")
    collections = relationship("Collection", back_populates="user")

    __table_args__ = (UniqueConstraint("name", "app_id", name="unique_name_app_user"),)

    def __repr__(self) -> str:
        return f"User(id={self.id}, app_id={self.app_id}, user_id={self.id}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    messages = relationship("Message", back_populates="session")
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    user = relationship("User", back_populates="sessions")

    def __repr__(self) -> str:
        return f"Session(id={self.id}, user_id={self.user_id}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.public_id"), index=True
    )
    is_user: Mapped[bool]
    content: Mapped[str] = mapped_column(String(65535))
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    metamessages = relationship("Metamessage", back_populates="message")

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, is_user={self.is_user}, content={self.content[10:]})"


class Metamessage(Base):
    __tablename__ = "metamessages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    metamessage_type: Mapped[str] = mapped_column(String(512), index=True)
    content: Mapped[str] = mapped_column(String(65535))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.public_id"), index=True
    )

    message = relationship("Message", back_populates="metamessages")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    def __repr__(self) -> str:
        return f"Metamessages(id={self.id}, message_id={self.message_id}, metamessage_type={self.metamessage_type}, content={self.content[10:]})"


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(String(512), index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    user = relationship("User", back_populates="collections")
    user_id: Mapped[str] = mapped_column(
        String(21), ForeignKey("users.public_id"), index=True
    )

    __table_args__ = (
        UniqueConstraint("name", "user_id", name="unique_name_collection_user"),
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        String(21), index=True, unique=True, default=generate_nanoid
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    content: Mapped[str] = mapped_column(String(65535))
    embedding = mapped_column(Vector(1536))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )

    collection_id: Mapped[str] = mapped_column(
        String(21), ForeignKey("collections.public_id"), index=True
    )
    collection = relationship("Collection", back_populates="documents")


class QueueItem(Base):
    __tablename__ = "queue"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)


class ActiveQueueSession(Base):
    __tablename__ = "active_queue_sessions"

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    last_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )
