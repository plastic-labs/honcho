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
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base

load_dotenv()


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    name: Mapped[str] = mapped_column(TEXT, index=True, unique=True)
    peers = relationship("Peer", back_populates="workspace")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
    )


class SessionPeer(Base):
    __tablename__ = "session_peers"
    session_public_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("sessions.public_id"), primary_key=True
    )
    peer_public_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("peers.public_id"), primary_key=True
    )


class Peer(Base):
    __tablename__ = "peers"
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
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.public_id"), index=True
    )
    workspace = relationship("Workspace", back_populates="peers")
    sessions = relationship("Session", secondary="session_peers", back_populates="peers")
    collections = relationship("Collection", back_populates="peer")

    __table_args__ = (
        UniqueConstraint("name", "workspace_id", name="unique_name_workspace_peer"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        Index("idx_peers_workspace_lookup", "workspace_id", "public_id"),
    )

    def __repr__(self) -> str:
        return f"Peer(id={self.id}, workspace_id={self.workspace_id}, public_id={self.public_id} created_at={self.created_at}, h_metadata={self.h_metadata})"


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
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.public_id"), index=True
    )
    peers = relationship("Peer", secondary="session_peers", back_populates="sessions")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, index=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.public_id"), index=True
    )
    content: Mapped[str] = mapped_column(TEXT)
    h_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default={})

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    sender_id: Mapped[str] = mapped_column(ForeignKey("peers.public_id"), index=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.public_id"), index=True
    )

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        Index(
            "idx_messages_session_lookup",
            "session_id",
            "id",
            postgresql_include=["public_id", "created_at"],
        ),
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_id={self.session_id}, content={self.content[10:]})"


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
    peer = relationship("Peer", back_populates="collections")
    peer_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("peers.public_id"), index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.public_id"), index=True
    )

    __table_args__ = (
        UniqueConstraint("name", "peer_id", name="unique_name_collection_peer"),
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
    peer_id: Mapped[str] = mapped_column(ForeignKey("peers.public_id"), index=True)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.public_id"), index=True
    )
    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        # HNSW index on embedding column
        Index(
            "idx_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",  # HNSW index type
            postgresql_with={"m": 16, "ef_construction": 64},  # HNSW parameters
            postgresql_ops={"embedding": "vector_cosine_ops"},  # Cosine distance operator
        ),
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
