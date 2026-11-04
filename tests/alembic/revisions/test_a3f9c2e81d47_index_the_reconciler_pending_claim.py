"""Hooks for revision a3f9c2e81d47 (tenant-scoped read indexes, and the indexes they replace)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

NEW = [
    ("documents", "ix_documents_pending_claim"),
    ("message_embeddings", "ix_message_embeddings_pending_claim"),
    ("sessions", "ix_sessions_tenant_workspace_created"),
    ("peers", "ix_peers_tenant_workspace_created"),
    ("documents", "ix_documents_tenant_collection_created"),
    ("documents", "ix_documents_tenant_session"),
    ("documents", "ix_documents_legacy_sources_tenant"),
    ("queue", "ix_queue_tenant_workspace"),
]
RETIRED = [
    ("documents", "ix_documents_sync_state_last_sync_at"),
    ("message_embeddings", "ix_message_embeddings_sync_state_last_sync_at"),
    ("sessions", "ix_sessions_tenant_workspace"),
    ("sessions", "ix_sessions_created_at"),
    ("peers", "ix_peers_tenant_workspace"),
    ("peers", "ix_peers_created_at"),
    ("documents", "ix_documents_tenant_collection"),
    ("documents", "ix_documents_created_at"),
    ("message_embeddings", "ix_message_embeddings_created_at"),
    ("documents", "ix_documents_legacy_sources_pending"),
    ("queue", "ix_queue_tenant_id"),
    ("messages", "ix_messages_session_lookup"),
]


@register_before_upgrade("a3f9c2e81d47")
def prepare_tenant_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(NEW)


@register_after_upgrade("a3f9c2e81d47")
def verify_tenant_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_valid(NEW)
    verifier.assert_indexes_not_exist(RETIRED)
    verifier.assert_indexes_valid([("documents", "ix_documents_deleted_at")])
    partial = verifier.conn.scalar(
        text(
            "SELECT indpred IS NOT NULL FROM pg_index"
            + " WHERE indexrelid = to_regclass(:name)"
        ),
        {"name": f'"{verifier.schema}".ix_documents_deleted_at'},
    )
    assert partial, "ix_documents_deleted_at must be partial after a3f9c2e81d47"
