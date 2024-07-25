import datetime
import uuid

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base

load_dotenv()

# DATABASE_TYPE = os.getenv("DATABASE_TYPE", "postgres")

# ColumnType = JSONB if DATABASE_TYPE == "postgres" else JSON


class App(Base):
    __tablename__ = "apps"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), index=True, unique=True)
    users = relationship("User", back_populates="app")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    # Add any additional fields for an app here


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), index=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("apps.id"), index=True)
    app = relationship("App", back_populates="users")
    sessions = relationship("Session", back_populates="user")
    collections = relationship("Collection", back_populates="user")

    __table_args__ = (UniqueConstraint("name", "app_id", name="unique_name_app_user"),)

    def __repr__(self) -> str:
        return f"User(id={self.id}, app_id={self.app_id}, user_id={self.user_id}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    location_id: Mapped[str] = mapped_column(String(512), index=True, default="default")
    is_active: Mapped[bool] = mapped_column(default=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    messages = relationship("Message", back_populates="session")
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    user = relationship("User", back_populates="sessions")

    def __repr__(self) -> str:
        return f"Session(id={self.id}, user_id={self.user_id}, location_id={self.location_id}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id"), index=True)
    is_user: Mapped[bool]
    content: Mapped[str] = mapped_column(String(65535))
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    metamessages = relationship("Metamessage", back_populates="message")

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, is_user={self.is_user}, content={self.content[10:]})"


class Metamessage(Base):
    __tablename__ = "metamessages"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    metamessage_type: Mapped[str] = mapped_column(String(512), index=True)
    content: Mapped[str] = mapped_column(String(65535))
    message_id = Column(Uuid, ForeignKey("messages.id"), index=True)

    message = relationship("Message", back_populates="metamessages")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    def __repr__(self) -> str:
        return f"Metamessages(id={self.id}, message_id={self.message_id}, metamessage_type={self.metamessage_type}, content={self.content[10:]})"


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(512), index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    user = relationship("User", back_populates="collections")
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)

    __table_args__ = (
        UniqueConstraint("name", "user_id", name="unique_name_collection_user"),
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, index=True, default=uuid.uuid4
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    content: Mapped[str] = mapped_column(String(65535))
    embedding = mapped_column(Vector(1536))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    collection_id = Column(Uuid, ForeignKey("collections.id"), index=True)
    collection = relationship("Collection", back_populates="documents")


class QueueItem(Base):
    __tablename__ = "queue"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
