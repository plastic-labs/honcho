import datetime
from logging import getLogger
from typing import Any

import tiktoken
from dotenv import load_dotenv
from nanoid import generate as generate_nanoid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    Table,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base

load_dotenv(override=True)

logger = getLogger(__name__)

# Initialize tiktoken encoder for token counting for message content
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        # Fallback: rough estimation (4 chars per token)
        logger.warning(
            f"Error counting tokens for text: {text[:50]}{'...' if len(text) > 50 else ''}, using fallback (4 chars per token). Error: {str(e)}"
        )
        return len(text) // 4


# Association table for many-to-many relationship between sessions and peers
session_peers_table = Table(
    "session_peers",
    Base.metadata,
    Column(
        "workspace_name",
        TEXT,
        ForeignKey("workspaces.name"),
        primary_key=True,
        nullable=False,
    ),
    Column(
        "session_name",
        TEXT,
        primary_key=True,
        nullable=False,
    ),
    Column("peer_name", TEXT, primary_key=True, nullable=False),
    Column("configuration", JSONB, default=dict),
    Column("internal_metadata", JSONB, default=dict),
    Column(
        "joined_at",
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
    ),
    Column(
        "left_at",
        DateTime(timezone=True),
        nullable=True,
    ),
    # Composite foreign key constraint for sessions
    ForeignKeyConstraint(
        ["session_name", "workspace_name"],
        ["sessions.name", "sessions.workspace_name"],
    ),
    # Composite foreign key constraint for peers
    ForeignKeyConstraint(
        ["peer_name", "workspace_name"],
        ["peers.name", "peers.workspace_name"],
    ),
)


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
    name: Mapped[str] = mapped_column(TEXT, index=True, unique=True)
    peers = relationship("Peer", back_populates="workspace")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
    )


class Peer(Base):
    __tablename__ = "peers"
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
    name: Mapped[str] = mapped_column(TEXT, index=True)
    h_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    workspace_name: Mapped[str] = mapped_column(
        ForeignKey("workspaces.name"), index=True
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    workspace = relationship("Workspace", back_populates="peers")
    sessions = relationship(
        "Session", secondary=session_peers_table, back_populates="peers"
    )
    collections = relationship("Collection", back_populates="peer")

    __table_args__ = (
        UniqueConstraint("name", "workspace_name", name="unique_name_workspace_peer"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        Index("idx_peers_workspace_lookup", "workspace_name", "name"),
    )

    def __repr__(self) -> str:
        return f"Peer(id={self.id}, name={self.name}, workspace_name={self.workspace_name}, created_at={self.created_at}, h_metadata={self.h_metadata}, configuration={self.configuration})"


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(TEXT, primary_key=True, default=generate_nanoid)
    name: Mapped[str] = mapped_column(TEXT, index=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    h_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    messages = relationship("Message", back_populates="session")
    workspace_name: Mapped[str] = mapped_column(
        ForeignKey("workspaces.name"), index=True
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    peers = relationship(
        "Peer", secondary=session_peers_table, back_populates="sessions"
    )

    __table_args__ = (
        UniqueConstraint("name", "workspace_name", name="unique_session_name"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, name={self.name}, workspace_name={self.workspace_name}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    public_id: Mapped[str] = mapped_column(
        TEXT, index=True, unique=True, default=generate_nanoid
    )
    session_name: Mapped[str | None] = mapped_column(index=True, nullable=True)
    content: Mapped[str] = mapped_column(TEXT)
    h_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    session = relationship("Session", back_populates="messages")
    peer_name: Mapped[str] = mapped_column(index=True)
    workspace_name: Mapped[str] = mapped_column(
        ForeignKey("workspaces.name"), index=True
    )

    __table_args__ = (
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        # Composite foreign key constraint for sessions
        ForeignKeyConstraint(
            ["session_name", "workspace_name"],
            ["sessions.name", "sessions.workspace_name"],
        ),
        # Composite foreign key constraint for peers
        ForeignKeyConstraint(
            ["peer_name", "workspace_name"],
            ["peers.name", "peers.workspace_name"],
        ),
        Index(
            "idx_messages_session_lookup",
            "session_name",
            "id",
            postgresql_include=["id", "created_at"],
        ),
        # Full text search index on content column
        Index(
            "idx_messages_content_gin",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, session_name={self.session_name}, peer_name={self.peer_name}, content={self.content})"


@event.listens_for(Message, "before_insert")
def calculate_token_count_on_insert(_mapper, _connection, target):
    """Calculate token count before inserting a new message."""
    target.token_count = count_tokens(target.content)


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
    name: Mapped[str] = mapped_column(TEXT, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )
    h_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    peer = relationship("Peer", back_populates="collections")
    peer_name: Mapped[str] = mapped_column(TEXT, index=True)
    workspace_name: Mapped[str] = mapped_column(
        ForeignKey("workspaces.name"), index=True
    )

    __table_args__ = (
        UniqueConstraint(
            "name", "peer_name", "workspace_name", name="unique_name_collection_peer"
        ),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        # Composite foreign key constraint for peers
        ForeignKeyConstraint(
            ["peer_name", "workspace_name"],
            ["peers.name", "peers.workspace_name"],
        ),
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict
    )
    content: Mapped[str] = mapped_column(TEXT)
    embedding = mapped_column(Vector(1536))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=func.now()
    )

    collection_name: Mapped[str] = mapped_column(TEXT, index=True)
    peer_name: Mapped[str] = mapped_column(index=True)
    workspace_name: Mapped[str] = mapped_column(
        ForeignKey("workspaces.name"), index=True
    )
    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        # Composite foreign key constraint for collections
        ForeignKeyConstraint(
            ["collection_name", "peer_name", "workspace_name"],
            ["collections.name", "collections.peer_name", "collections.workspace_name"],
        ),
        # Composite foreign key constraint for peers
        ForeignKeyConstraint(
            ["peer_name", "workspace_name"],
            ["peers.name", "peers.workspace_name"],
        ),
        # HNSW index on embedding column
        Index(
            "idx_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",  # HNSW index type
            postgresql_with={"m": 16, "ef_construction": 64},  # HNSW parameters
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            },  # Cosine distance operator
        ),
    )


class QueueItem(Base):
    __tablename__ = "queue"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id"), index=True, nullable=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)


class ActiveQueueSession(Base):
    __tablename__ = "active_queue_sessions"

    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True
    )
    sender_name: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    target_name: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    task_type: Mapped[str] = mapped_column(TEXT)
    last_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sender_name",
            "target_name",
            "task_type",
            name="unique_active_queue_session",
        ),
    )


class SessionPeer(Base):
    __table__ = session_peers_table
