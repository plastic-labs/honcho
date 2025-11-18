"""standardize_constraint_names

Revision ID: baa22cad81e2
Revises: b8183c5ffb48
Create Date: 2025-11-15 01:16:40.937103

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

from migrations.utils import constraint_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "baa22cad81e2"
down_revision: str | None = "29ade7350c19"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # ============================================================
    # CONSTRAINT RENAMES (using raw SQL)
    # ============================================================

    # Unique Constraints
    if constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique", inspector
    ):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.active_queue_sessions RENAME CONSTRAINT "unique_work_unit_key" TO "uq_active_queue_sessions_work_unit_key"'
            )
        )

    if constraint_exists(
        "collections", "unique_observer_observed_collection", "unique", inspector
    ):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.collections RENAME CONSTRAINT "unique_observer_observed_collection" TO "uq_collections_observer_observed_workspace_name"'
            )
        )

    if constraint_exists("peers", "unique_name_workspace_peer", "unique", inspector):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.peers RENAME CONSTRAINT "unique_name_workspace_peer" TO "uq_peers_name_workspace_name"'
            )
        )

    if constraint_exists("sessions", "unique_session_name", "unique", inspector):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.sessions RENAME CONSTRAINT "unique_session_name" TO "uq_sessions_name_workspace_name"'
            )
        )

    if constraint_exists("messages", "uq_messages_session_seq", "unique", inspector):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.messages RENAME CONSTRAINT "uq_messages_session_seq" TO "uq_messages_workspace_name_session_name_seq_in_session"'
            )
        )

    if constraint_exists("workspaces", "uq_apps_name", "unique", inspector):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.workspaces RENAME CONSTRAINT "uq_apps_name" TO "uq_workspaces_name"'
            )
        )

    # Foreign Keys
    if constraint_exists(
        "message_embeddings",
        "message_embeddings_message_id_fkey",
        "foreignkey",
        inspector,
    ):
        conn.execute(
            text(
                f'ALTER TABLE {schema}.message_embeddings RENAME CONSTRAINT "message_embeddings_message_id_fkey" TO "fk_message_embeddings_message_id_messages"'
            )
        )

    # ============================================================
    # INDEX RENAMES (using raw SQL)
    # ============================================================

    # Collections
    if index_exists("collections", "idx_collections_observed", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_collections_observed" RENAME TO "ix_collections_observed"'
            )
        )
    if index_exists("collections", "idx_collections_observer", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_collections_observer" RENAME TO "ix_collections_observer"'
            )
        )

    # Documents
    if index_exists("documents", "idx_documents_embedding_hnsw", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_documents_embedding_hnsw" RENAME TO "ix_documents_embedding_hnsw"'
            )
        )
    if index_exists("documents", "idx_documents_observed", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_documents_observed" RENAME TO "ix_documents_observed"'
            )
        )
    if index_exists("documents", "idx_documents_observer", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_documents_observer" RENAME TO "ix_documents_observer"'
            )
        )
    if index_exists("documents", "idx_documents_session_name", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_documents_session_name" RENAME TO "ix_documents_session_name"'
            )
        )

    # Message Embeddings
    if index_exists(
        "message_embeddings", "idx_message_embeddings_created_at", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_created_at" RENAME TO "ix_message_embeddings_created_at"'
            )
        )
    if index_exists(
        "message_embeddings", "idx_message_embeddings_embedding_hnsw", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_embedding_hnsw" RENAME TO "ix_message_embeddings_embedding_hnsw"'
            )
        )
    if index_exists(
        "message_embeddings", "idx_message_embeddings_message_id", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_message_id" RENAME TO "ix_message_embeddings_message_id"'
            )
        )
    if index_exists(
        "message_embeddings", "idx_message_embeddings_peer_name", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_peer_name" RENAME TO "ix_message_embeddings_peer_name"'
            )
        )
    if index_exists(
        "message_embeddings", "idx_message_embeddings_session_name", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_session_name" RENAME TO "ix_message_embeddings_session_name"'
            )
        )
    if index_exists(
        "message_embeddings", "idx_message_embeddings_workspace_name", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_message_embeddings_workspace_name" RENAME TO "ix_message_embeddings_workspace_name"'
            )
        )

    # Messages
    if index_exists("messages", "idx_messages_content_gin", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_messages_content_gin" RENAME TO "ix_messages_content_gin"'
            )
        )
    if index_exists("messages", "idx_messages_session_lookup", inspector):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_messages_session_lookup" RENAME TO "ix_messages_session_lookup"'
            )
        )

    # Webhook Endpoints
    if index_exists(
        "webhook_endpoints", "idx_webhook_endpoints_workspace_lookup", inspector
    ):
        conn.execute(
            text(
                f'ALTER INDEX {schema}."idx_webhook_endpoints_workspace_lookup" RENAME TO "ix_webhook_endpoints_workspace_name"'
            )
        )

    # ============================================================
    # INDEX DELETIONS (redundant indexes being removed)
    # ============================================================

    # Drop redundant indexes on primary keys and unique columns
    if index_exists("messages", "ix_messages_id", inspector):
        op.drop_index("ix_messages_id", table_name="messages", schema=schema)
    if index_exists("messages", "ix_messages_public_id", inspector):
        op.drop_index("ix_messages_public_id", table_name="messages", schema=schema)
    if index_exists("peers", "ix_peers_name", inspector):
        op.drop_index("ix_peers_name", table_name="peers", schema=schema)
    if index_exists("peers", "idx_peers_workspace_lookup", inspector):
        op.drop_index("idx_peers_workspace_lookup", table_name="peers", schema=schema)
    if index_exists("workspaces", "ix_workspaces_name", inspector):
        op.drop_index("ix_workspaces_name", table_name="workspaces", schema=schema)

    # Drop old queue indexes that will be replaced
    if index_exists("queue", "ix_queue_work_unit_key_processed_id", inspector):
        op.drop_index(
            "ix_queue_work_unit_key_processed_id", table_name="queue", schema=schema
        )
    if index_exists("queue", "ix_queue_workspace_name_processed", inspector):
        op.drop_index(
            "ix_queue_workspace_name_processed", table_name="queue", schema=schema
        )

    # ============================================================
    # NEW INDEXES (being created, not renamed)
    # ============================================================

    # Queue - new indexes
    if not index_exists("queue", "ix_queue_processed", inspector):
        op.create_index(
            "ix_queue_processed", "queue", ["processed"], unique=False, schema=schema
        )
    if not index_exists("queue", "work_unit_key", inspector):
        op.create_index(
            "work_unit_key", "queue", ["processed", "id"], unique=False, schema=schema
        )

    # Sessions - new index
    if not index_exists("sessions", "ix_sessions_workspace_name", inspector):
        op.create_index(
            "ix_sessions_workspace_name",
            "sessions",
            ["workspace_name"],
            unique=False,
            schema=schema,
        )

    # Messages - drop FK that's being removed
    if constraint_exists(
        "messages", "fk_messages_workspace_name_workspaces", "foreignkey", inspector
    ):
        op.drop_constraint(
            "fk_messages_workspace_name_workspaces",
            "messages",
            schema=schema,
            type_="foreignkey",
        )


def downgrade() -> None:
    conn = op.get_bind()

    # ============================================================
    # REVERSE: NEW INDEXES AND FKS
    # ============================================================

    # Recreate FK
    op.create_foreign_key(
        "fk_messages_workspace_name_workspaces",
        "messages",
        "workspaces",
        ["workspace_name"],
        ["name"],
        source_schema=schema,
        referent_schema=schema,
    )

    # Drop new indexes
    op.drop_index("ix_sessions_workspace_name", table_name="sessions", schema=schema)
    op.drop_index("work_unit_key", table_name="queue", schema=schema)
    op.drop_index("ix_queue_processed", table_name="queue", schema=schema)

    # ============================================================
    # REVERSE: INDEX DELETIONS (recreate them)
    # ============================================================

    op.create_index(
        "ix_queue_workspace_name_processed",
        "queue",
        ["workspace_name", "processed"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_queue_work_unit_key_processed_id",
        "queue",
        ["work_unit_key", "processed", "id"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_workspaces_name", "workspaces", ["name"], unique=False, schema=schema
    )
    op.create_index(
        "idx_peers_workspace_lookup",
        "peers",
        ["workspace_name", "name"],
        unique=False,
        schema=schema,
    )
    op.create_index("ix_peers_name", "peers", ["name"], unique=False, schema=schema)
    op.create_index(
        "ix_messages_public_id", "messages", ["public_id"], unique=False, schema=schema
    )
    op.create_index("ix_messages_id", "messages", ["id"], unique=False, schema=schema)

    # ============================================================
    # REVERSE: INDEX RENAMES (using raw SQL)
    # ============================================================

    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_webhook_endpoints_workspace_name" RENAME TO "idx_webhook_endpoints_workspace_lookup"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_messages_session_lookup" RENAME TO "idx_messages_session_lookup"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_messages_content_gin" RENAME TO "idx_messages_content_gin"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_workspace_name" RENAME TO "idx_message_embeddings_workspace_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_session_name" RENAME TO "idx_message_embeddings_session_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_peer_name" RENAME TO "idx_message_embeddings_peer_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_message_id" RENAME TO "idx_message_embeddings_message_id"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_embedding_hnsw" RENAME TO "idx_message_embeddings_embedding_hnsw"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_message_embeddings_created_at" RENAME TO "idx_message_embeddings_created_at"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_documents_session_name" RENAME TO "idx_documents_session_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_documents_observer" RENAME TO "idx_documents_observer"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_documents_observed" RENAME TO "idx_documents_observed"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_documents_embedding_hnsw" RENAME TO "idx_documents_embedding_hnsw"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_collections_observer" RENAME TO "idx_collections_observer"'
        )
    )
    conn.execute(
        text(
            f'ALTER INDEX {schema}."ix_collections_observed" RENAME TO "idx_collections_observed"'
        )
    )

    # ============================================================
    # REVERSE: CONSTRAINT RENAMES
    # ============================================================

    # Foreign Keys
    conn.execute(
        text(
            f'ALTER TABLE {schema}.message_embeddings RENAME CONSTRAINT "fk_message_embeddings_message_id_messages" TO "message_embeddings_message_id_fkey"'
        )
    )

    # Unique Constraints
    conn.execute(
        text(
            f'ALTER TABLE {schema}.workspaces RENAME CONSTRAINT "uq_workspaces_name" TO "uq_apps_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE {schema}.messages RENAME CONSTRAINT "uq_messages_workspace_name_session_name_seq_in_session" TO "uq_messages_session_seq"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE {schema}.sessions RENAME CONSTRAINT "uq_sessions_name_workspace_name" TO "unique_session_name"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE {schema}.peers RENAME CONSTRAINT "uq_peers_name_workspace_name" TO "unique_name_workspace_peer"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE {schema}.collections RENAME CONSTRAINT "uq_collections_observer_observed_workspace_name" TO "unique_observer_observed_collection"'
        )
    )
    conn.execute(
        text(
            f'ALTER TABLE {schema}.active_queue_sessions RENAME CONSTRAINT "uq_active_queue_sessions_work_unit_key" TO "unique_work_unit_key"'
        )
    )
