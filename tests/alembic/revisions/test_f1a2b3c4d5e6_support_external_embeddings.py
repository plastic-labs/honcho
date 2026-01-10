"""Hooks for revision f1a2b3c4d5e6 (support_external_embeddings)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Indexes created by this migration
INDEXES = (
    ("documents", "ix_documents_deleted_at"),
    ("documents", "ix_documents_sync_state"),
    ("documents", "ix_documents_sync_state_last_sync_at"),
    ("message_embeddings", "ix_message_embeddings_sync_state"),
    ("message_embeddings", "ix_message_embeddings_sync_state_last_sync_at"),
)


@register_before_upgrade("f1a2b3c4d5e6")
def prepare_support_external_embeddings(
    verifier: MigrationVerifier,
) -> None:
    """Seed state and assertions before upgrading to f1a2b3c4d5e6."""
    # Embedding columns should be NOT NULL before migration
    verifier.assert_column_exists("message_embeddings", "embedding", nullable=False)
    verifier.assert_column_exists("documents", "embedding", nullable=False)

    # Soft delete column should not exist
    verifier.assert_column_exists("documents", "deleted_at", exists=False)

    # Sync state columns should not exist on documents
    verifier.assert_column_exists("documents", "sync_state", exists=False)
    verifier.assert_column_exists("documents", "last_sync_at", exists=False)
    verifier.assert_column_exists("documents", "sync_attempts", exists=False)

    # Sync state columns should not exist on message_embeddings
    verifier.assert_column_exists("message_embeddings", "sync_state", exists=False)
    verifier.assert_column_exists("message_embeddings", "last_sync_at", exists=False)
    verifier.assert_column_exists("message_embeddings", "sync_attempts", exists=False)

    # Indexes should not exist
    verifier.assert_indexes_not_exist(INDEXES)


@register_after_upgrade("f1a2b3c4d5e6")
def verify_support_external_embeddings(
    verifier: MigrationVerifier,
) -> None:
    """Add assertions validating the effects of f1a2b3c4d5e6."""
    # Embedding columns should now be nullable
    verifier.assert_column_exists("message_embeddings", "embedding", nullable=True)
    verifier.assert_column_exists("documents", "embedding", nullable=True)

    # Soft delete column should exist and be nullable
    verifier.assert_column_exists("documents", "deleted_at", nullable=True)

    # Sync state columns should exist on documents
    verifier.assert_column_exists("documents", "sync_state", nullable=False)
    verifier.assert_column_exists("documents", "last_sync_at", nullable=True)
    verifier.assert_column_exists("documents", "sync_attempts", nullable=False)

    # Sync state columns should exist on message_embeddings
    verifier.assert_column_exists("message_embeddings", "sync_state", nullable=False)
    verifier.assert_column_exists("message_embeddings", "last_sync_at", nullable=True)
    verifier.assert_column_exists("message_embeddings", "sync_attempts", nullable=False)

    # All indexes should exist
    verifier.assert_indexes_exist(INDEXES)
