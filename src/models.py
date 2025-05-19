import datetime

from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
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
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True, unique=True)
    users = relationship("User", back_populates="app")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
    )


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    app = relationship("App", back_populates="users")
    sessions = relationship("Session", back_populates="user")
    collections = relationship("Collection", back_populates="user")
    metamessages = relationship("Metamessage", back_populates="user")

    __table_args__ = (
        UniqueConstraint("name", "app_id", name="unique_name_app_user"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        Index("idx_users_app_lookup", "app_id", "public_id"),
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, app_id={self.app_id}, public_id={self.public_id} created_at={self.created_at}, h_metadata={self.h_metadata})"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    messages = relationship("Message", back_populates="session")
    metamessages = relationship("Metamessage", back_populates="session")
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        Index("idx_sessions_user_lookup", "user_id", "public_id"),
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, user_id={self.user_id}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.public_id"), index=True
    )
    is_user: Mapped[bool]
    content: Mapped[str] = mapped_column(TEXT)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    metamessages = relationship("Metamessage", back_populates="message")
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        Index(
            "idx_messages_session_lookup",
            "session_id",
            "id",
            postgresql_include=["public_id", "is_user", "created_at"],
        ),
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, is_user={self.is_user}, content={self.content[10:]})"


class Metamessage(Base):
    __tablename__ = "metamessages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    label: Mapped[str] = mapped_column(TEXT, index=True)
    content: Mapped[str] = mapped_column(TEXT)

    # Foreign keys - message_id is now optional
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.public_id"), index=True, nullable=True
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.public_id"), index=True, nullable=True
    )

    # Relationships
    user = relationship("User", back_populates="metamessages")
    session = relationship("Session", back_populates="metamessages")
    message = relationship("Message", back_populates="metamessages")

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("length(label) <= 512", name="label_length"),
        # Added constraints to ensure consistency
        CheckConstraint(
            "(message_id IS NULL) OR (session_id IS NOT NULL)",
            name="message_requires_session",
        ),
        # Keep existing index
        Index(
            "idx_metamessages_lookup",
            "label",
            text("id DESC"),
            postgresql_include=["public_id", "message_id", "created_at"],
        ),
        # Indices for user, session, and message lookups
        Index(
            "idx_metamessages_user_lookup",
            "user_id",
            "label",
            text("id DESC"),
        ),
        Index(
            "idx_metamessages_session_lookup",
            "session_id",
            "label",
            text("id DESC"),
        ),
        Index(
            "idx_metamessages_message_lookup",
            "message_id",
            "label",
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:
        return f"Metamessages(id={self.id}, user_id={self.user_id}, session_id={self.session_id}, message_id={self.message_id}, label={self.label})"


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    user = relationship("User", back_populates="collections")
    user_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("users.public_id"), index=True
    )
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)

    __table_args__ = (
        UniqueConstraint("name", "user_id", name="unique_name_collection_user"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(name) <= 512", name="name_length"),
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    content: Mapped[str] = mapped_column(TEXT)
    embedding = mapped_column(Vector(1536))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )

    collection_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("collections.public_id"), index=True
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.public_id"), index=True)
    app_id: Mapped[str] = mapped_column(ForeignKey("apps.public_id"), index=True)
    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        # HNSW index on embedding column
        Index(
            "idx_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw", # HNSW index type
            postgresql_with={"m": 16, "ef_construction": 64}, # HNSW parameters
            postgresql_ops={"embedding": "vector_cosine_ops"}, # Cosine distance operator
        ),
        Index(
            "idx_documents_query_documents_lookup",
            "app_id",
            "user_id",
            "collection_id",
            unique=False
        )
    )


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
