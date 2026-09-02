import datetime
from logging import getLogger
from typing import Any, final, override

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
    PrimaryKeyConstraint,
    Table,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TEXT
from sqlalchemy.orm import Mapped, MappedColumn, mapped_column, relationship
from sqlalchemy.sql import func

from src.config import settings
from src.utils.types import DocumentLevel, TaskType, VectorSyncState

from .db import Base

load_dotenv(override=True)

_VECTOR_DIM: int = settings.EMBEDDING.VECTOR_DIMENSIONS

logger = getLogger(__name__)


# Association table for many-to-many relationship between sessions and peers
session_peers_table = Table(
    "session_peers",
    Base.metadata,
    # ai: tenant_id leads the all-natural-key PK and is the HASH partition key.
    Column(
        "tenant_id",
        TEXT,
        ForeignKey("tenants.tenant_id"),
        primary_key=True,
        nullable=False,
    ),
    Column(
        "workspace_name",
        TEXT,
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
    Column(
        "configuration",
        JSONB,
        default=dict,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    ),
    Column(
        "internal_metadata",
        JSONB,
        default=dict,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    ),
    Column(
        "joined_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Column(
        "left_at",
        DateTime(timezone=True),
        nullable=True,
    ),
    # Composite foreign key constraint for workspaces
    ForeignKeyConstraint(
        ["workspace_name", "tenant_id"],
        ["workspaces.name", "workspaces.tenant_id"],
    ),
    # Composite foreign key constraint for sessions
    ForeignKeyConstraint(
        ["session_name", "workspace_name", "tenant_id"],
        ["sessions.name", "sessions.workspace_name", "sessions.tenant_id"],
    ),
    # Composite foreign key constraint for peers
    ForeignKeyConstraint(
        ["peer_name", "workspace_name", "tenant_id"],
        ["peers.name", "peers.workspace_name", "peers.tenant_id"],
    ),
    postgresql_partition_by="HASH (tenant_id)",
)


@final
class Tenant(Base):
    """One row per tenant; the FK target for every tenant-scoped table's ``tenant_id``."""

    # region ai
    # honcho's data plane and the control plane that owns the canonical tenant
    # registry live in separate databases, so a cross-database FK to the real
    # source of truth is impossible; this table is honcho's local mirror, kept in
    # sync from the control plane (backfilled for existing tenants; written on
    # provision, or upserted on first authenticated request, for new ones). It is
    # also the one-row-per-tenant home for facts with nowhere else to live:
    #   - legacy_app_name: the tenant's original per-instance name. After the
    #     groudon shared-pool allocation, app_name is "shared" for every pooled
    #     tenant, so this column preserves the per-tenant value that keeps a
    #     tenant's external vector-store namespace stable across the move (avoids a
    #     full, expensive re-embed). Named "legacy_" so it never competes with the
    #     pool's app_name.
    #   - tier: whether the tenant runs on a dedicated or a shared backend.
    # endregion
    __tablename__: str = "tenants"
    tenant_id: Mapped[str] = mapped_column(TEXT, primary_key=True)
    legacy_app_name: Mapped[str | None] = mapped_column(TEXT, nullable=True, index=True)
    tier: Mapped[str] = mapped_column(TEXT, nullable=False, server_default="dedicated")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


@final
class Workspace(Base):
    __tablename__: str = "workspaces"
    # region ai
    # tenant_id is the HASH partition key and leads the composite PK, so it is
    # declared first. It FKs to the local tenants mirror (see Tenant).
    # endregion
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    name: Mapped[str] = mapped_column(TEXT)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    h_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    sessions = relationship(
        "Session", back_populates="workspace", cascade="all, delete, delete-orphan"
    )
    peers = relationship(
        "Peer", back_populates="workspace", cascade="all, delete, delete-orphan"
    )
    webhook_endpoints = relationship("WebhookEndpoint", back_populates="workspace")

    __table_args__ = (
        # region ai
        # Partitioned by HASH(tenant_id): Postgres requires the partition key in
        # the PK and in every UNIQUE. `name` is unique WITHIN a tenant, not
        # globally — many tenants share the SDK-default "default" workspace.
        # endregion
        PrimaryKeyConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "name"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )


@final
class Peer(Base):
    __tablename__: str = "peers"
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    h_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # region ai
    # workspace_name's FK to workspaces is now the composite (below), since
    # workspaces.name is only unique within a tenant.
    # endregion
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    workspace = relationship("Workspace", back_populates="peers")
    sessions = relationship(
        "Session", secondary=session_peers_table, back_populates="peers"
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "name", "workspace_name"),
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        Index("ix_peers_tenant_workspace", "tenant_id", "workspace_name"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )

    def __repr__(self) -> str:
        return f"Peer(tenant_id={self.tenant_id}, id={self.id}, name={self.name}, workspace_name={self.workspace_name}, created_at={self.created_at}, h_metadata={self.h_metadata}, configuration={self.configuration})"


@final
class Session(Base):
    __tablename__: str = "sessions"
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    name: Mapped[str] = mapped_column(TEXT)
    is_active: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    h_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    workspace = relationship("Workspace", back_populates="sessions")
    peers = relationship(
        "Peer", secondary=session_peers_table, back_populates="sessions"
    )
    messages = relationship("Message", back_populates="session")

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "name", "workspace_name"),
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        Index("ix_sessions_tenant_workspace", "tenant_id", "workspace_name"),
        CheckConstraint("length(name) <= 512", name="name_length"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )

    def __repr__(self) -> str:
        return f"Session(tenant_id={self.tenant_id}, id={self.id}, name={self.name}, workspace_name={self.workspace_name}, is_active={self.is_active}, created_at={self.created_at}, h_metadata={self.h_metadata})"


@final
class Message(Base):
    __tablename__: str = "messages"
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[int] = mapped_column(BigInteger, Identity(), autoincrement=True)
    public_id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    # NOTE: Messages in Honcho 2.0 could historically be stored outside of a session.
    # We have since assigned all of these messages to a default session.
    session_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    content: Mapped[str] = mapped_column(TEXT)
    h_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    seq_in_session: Mapped[int] = mapped_column(BigInteger, nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # Note: Foreign key relationships established via composite ForeignKeyConstraint below
    peer_name: Mapped[str] = mapped_column(TEXT)
    workspace_name: Mapped[str] = mapped_column(TEXT)

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        # ai: (tenant_id, public_id) is the unique that message_embeddings' FK targets.
        UniqueConstraint("tenant_id", "public_id"),
        CheckConstraint("length(public_id) = 21", name="public_id_length"),
        CheckConstraint("public_id ~ '^[A-Za-z0-9_-]+$'", name="public_id_format"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        # Composite foreign key constraint for sessions
        ForeignKeyConstraint(
            ["session_name", "workspace_name", "tenant_id"],
            ["sessions.name", "sessions.workspace_name", "sessions.tenant_id"],
        ),
        # Composite foreign key constraint for peers
        ForeignKeyConstraint(
            ["peer_name", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        Index(
            "ix_messages_session_lookup",
            "tenant_id",
            "session_name",
            "id",
            postgresql_include=["created_at"],
        ),
        Index(
            "ix_messages_peer_lookup",
            "tenant_id",
            "workspace_name",
            "peer_name",
            "created_at",
        ),
        UniqueConstraint(
            "tenant_id",
            "workspace_name",
            "session_name",
            "seq_in_session",
        ),
        # region ai
        # GIN can't lead with a scalar column without btree_gin; the table is
        # HASH(tenant_id)-partitioned, so this index is per-partition — queries
        # prune to one partition, then tenant_id filters the FTS candidates.
        # endregion
        Index(
            "ix_messages_content_gin",
            text("to_tsvector('english', content)"),
            postgresql_using="gin",
        ),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )

    @override
    def __repr__(self) -> str:
        return f"Message(tenant_id={self.tenant_id}, id={self.id}, session_name={self.session_name}, peer_name={self.peer_name}, content={self.content})"


@final
class MessageEmbedding(Base):
    __tablename__: str = "message_embeddings"

    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[int] = mapped_column(BigInteger, Identity(), autoincrement=True)
    content: Mapped[str] = mapped_column(TEXT)
    embedding: MappedColumn[Any] = mapped_column(Vector(_VECTOR_DIM), nullable=True)
    message_id: Mapped[str] = mapped_column(TEXT, nullable=False)
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    session_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    peer_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # Vector sync state tracking
    sync_state: Mapped[VectorSyncState] = mapped_column(
        TEXT, nullable=False, server_default="pending"
    )
    last_sync_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sync_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        # region ai
        # message_id → messages.public_id is now composite: messages' unique is
        # (tenant_id, public_id) under partitioning.
        # endregion
        ForeignKeyConstraint(
            ["tenant_id", "message_id"],
            ["messages.tenant_id", "messages.public_id"],
            ondelete="CASCADE",
        ),
        # region ai
        # Composite FK to workspaces, for parity with the other tenant-scoped
        # tables (workspace_name is only unique within a tenant).
        # endregion
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        ForeignKeyConstraint(
            ["session_name", "workspace_name", "tenant_id"],
            ["sessions.name", "sessions.workspace_name", "sessions.tenant_id"],
        ),
        ForeignKeyConstraint(
            ["peer_name", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        # region ai
        # message_id-leading: every lookup on message_id is cross-tenant (the
        # reconciler / embed_now filter by message_id with no tenant_id in scope),
        # so a tenant_id prefix would force a scan of all partitions.
        # endregion
        Index("ix_message_embeddings_message_tenant", "message_id", "tenant_id"),
        # region ai
        # HNSW is a single-column vector index (can't lead with tenant_id); it
        # becomes per-partition automatically under HASH(tenant_id).
        # endregion
        Index(
            "ix_message_embeddings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # region ai
        # NOT tenant_id-leading on purpose: the reconciler scans this cross-tenant
        # (sync_state='pending' over all tenants), so a tenant_id prefix wouldn't
        # help. (Also drops the redundant single-column sync_state index.)
        # endregion
        Index(
            "ix_message_embeddings_sync_state_last_sync_at",
            "sync_state",
            "last_sync_at",
        ),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )


@final
class Collection(Base):
    __tablename__: str = "collections"

    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    observer: Mapped[str] = mapped_column(TEXT)
    observed: Mapped[str] = mapped_column(TEXT)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    h_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    documents = relationship(
        "Document", back_populates="collection", cascade="all, delete, delete-orphan"
    )
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        UniqueConstraint(
            "tenant_id",
            "observer",
            "observed",
            "workspace_name",
        ),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        # Composite foreign key constraint for observer peer
        ForeignKeyConstraint(
            ["observer", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        # Composite foreign key constraint for observed peer
        ForeignKeyConstraint(
            ["observed", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )


@final
class Document(Base):
    __tablename__: str = "documents"
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    internal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "internal_metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    content: Mapped[str] = mapped_column(TEXT)
    level: Mapped[DocumentLevel] = mapped_column(
        TEXT, nullable=False, server_default="explicit"
    )
    times_derived: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    embedding: MappedColumn[Any] = mapped_column(Vector(_VECTOR_DIM), nullable=True)
    source_ids: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, server_default=text("NULL")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    observer: Mapped[str] = mapped_column(TEXT)
    observed: Mapped[str] = mapped_column(TEXT)
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    session_name: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True, default=None
    )

    # Vector sync state tracking
    sync_state: Mapped[VectorSyncState] = mapped_column(
        TEXT, nullable=False, server_default="pending"
    )
    last_sync_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sync_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    collection = relationship("Collection", back_populates="documents")

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        CheckConstraint("length(id) = 21", name="id_length"),
        CheckConstraint("length(content) <= 65535", name="content_length"),
        CheckConstraint("id ~ '^[A-Za-z0-9_-]+$'", name="id_format"),
        # Composite foreign key constraint for workspaces
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        # Composite foreign key constraint for collections
        ForeignKeyConstraint(
            ["observer", "observed", "workspace_name", "tenant_id"],
            [
                "collections.observer",
                "collections.observed",
                "collections.workspace_name",
                "collections.tenant_id",
            ],
        ),
        # Composite foreign key constraint for observer peer
        ForeignKeyConstraint(
            ["observer", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        # Composite foreign key constraint for observed peer
        ForeignKeyConstraint(
            ["observed", "workspace_name", "tenant_id"],
            ["peers.name", "peers.workspace_name", "peers.tenant_id"],
        ),
        # Composite foreign key constraint for sessions
        ForeignKeyConstraint(
            ["session_name", "workspace_name", "tenant_id"],
            ["sessions.name", "sessions.workspace_name", "sessions.tenant_id"],
        ),
        # Tenant-scoped collection lookups
        # ai: replaces the single observer/observed indexes
        Index(
            "ix_documents_tenant_collection",
            "tenant_id",
            "observer",
            "observed",
            "workspace_name",
        ),
        # ai: HNSW is a single-column vector index (per-partition under HASH(tenant_id))
        Index(
            "ix_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",  # HNSW index type
            postgresql_with={"m": 16, "ef_construction": 64},  # HNSW parameters
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            },  # Cosine distance operator
        ),
        # GIN index for efficient tree traversal (finding children by source IDs)
        Index(
            "ix_documents_source_ids_gin",
            "source_ids",
            postgresql_using="gin",
        ),
        # region ai
        # Reconciler scans this cross-tenant (sync_state='pending'), so NOT
        # tenant_id-leading. Also drops the redundant single-column sync_state index.
        # endregion
        Index(
            "ix_documents_sync_state_last_sync_at",
            "sync_state",
            "last_sync_at",
        ),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )


@final
class QueueItem(Base):
    __tablename__: str = "queue"
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    # region ai
    # Service table: NOT partitioned and drained-not-copied at migration, so it
    # keeps a sole-id PK. tenant_id is a plain attribution / fair-scheduling column
    # (no FK, no RLS); the FKs to the now-partitioned sessions / messages /
    # workspaces are dropped — the app manages queue lifecycle and already
    # tolerates missing referents.
    # endregion
    tenant_id: Mapped[str | None] = mapped_column(TEXT, nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(TEXT, nullable=True, index=True)
    work_unit_key: Mapped[str] = mapped_column(TEXT, nullable=False)

    task_type: Mapped[TaskType] = mapped_column(TEXT, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    processed: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), index=True
    )
    error: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    workspace_name: Mapped[str | None] = mapped_column(TEXT, nullable=True, index=True)
    message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        Index(
            "ix_queue_message_id_not_null",
            "message_id",
            postgresql_where=text("message_id IS NOT NULL"),
        ),
        Index(
            "ix_queue_work_unit_key_processed_id",
            "work_unit_key",
            "processed",
            "id",
        ),
        # Partial unique index for reconciler task deduplication
        Index(
            "uq_queue_reconciler_pending_work_unit_key",
            "work_unit_key",
            unique=True,
            postgresql_where=text("task_type = 'reconciler' AND processed = false"),
        ),
        # Partial unique index for dream task deduplication
        Index(
            "uq_queue_dream_pending_work_unit_key",
            "work_unit_key",
            unique=True,
            postgresql_where=text("task_type = 'dream' AND processed = false"),
        ),
    )

    def __repr__(self) -> str:
        return f"QueueItem(id={self.id}, tenant_id={self.tenant_id}, session_id={self.session_id}, work_unit_key={self.work_unit_key}, task_type={self.task_type}, payload={self.payload}, processed={self.processed}, workspace_name={self.workspace_name}, message_id={self.message_id})"


@final
class ActiveQueueSession(Base):
    __tablename__: str = "active_queue_sessions"

    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid, primary_key=True)

    # ai: Service table (unpartitioned): tenant_id is plain attribution, no FK / RLS.
    tenant_id: Mapped[str | None] = mapped_column(TEXT, nullable=True)

    work_unit_key: Mapped[str] = mapped_column(TEXT, unique=True)

    last_updated: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


@final
class WebhookEndpoint(Base):
    __tablename__: str = "webhook_endpoints"
    tenant_id: Mapped[str] = mapped_column(
        TEXT, ForeignKey("tenants.tenant_id"), nullable=False
    )
    id: Mapped[str] = mapped_column(TEXT, default=generate_nanoid)
    workspace_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    url: Mapped[str] = mapped_column(TEXT, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    workspace = relationship("Workspace", back_populates="webhook_endpoints")

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["workspace_name", "tenant_id"],
            ["workspaces.name", "workspaces.tenant_id"],
        ),
        Index("ix_webhook_endpoints_tenant_workspace", "tenant_id", "workspace_name"),
        CheckConstraint("length(url) <= 2048", name="url_length"),
        {"postgresql_partition_by": "HASH (tenant_id)"},
    )

    def __repr__(self) -> str:
        return f"WebhookEndpoint(tenant_id={self.tenant_id}, id={self.id}, workspace_name={self.workspace_name}, url={self.url})"


@final
class SessionPeer(Base):
    __table__: Table = session_peers_table

    # Type annotations for the columns
    tenant_id: Mapped[str]
    workspace_name: Mapped[str]
    session_name: Mapped[str]
    peer_name: Mapped[str]
    configuration: Mapped[dict[str, Any]]
    internal_metadata: Mapped[dict[str, Any]]
    joined_at: Mapped[datetime.datetime]
    left_at: Mapped[datetime.datetime | None]
