"""Hooks for revision 917195d9b5e9 (message embeddings table)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEXES = (
    ("message_embeddings", "idx_message_embeddings_message_id"),
    ("message_embeddings", "idx_message_embeddings_workspace_name"),
    ("message_embeddings", "idx_message_embeddings_session_name"),
    ("message_embeddings", "idx_message_embeddings_peer_name"),
    ("message_embeddings", "idx_message_embeddings_created_at"),
    ("message_embeddings", "idx_message_embeddings_embedding_hnsw"),
)


@register_before_upgrade("917195d9b5e9")
def prepare_message_embeddings(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("message_embeddings", exists=False)


@register_after_upgrade("917195d9b5e9")
def verify_message_embeddings_table(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("message_embeddings")
    verifier.assert_indexes_exist(INDEXES)
