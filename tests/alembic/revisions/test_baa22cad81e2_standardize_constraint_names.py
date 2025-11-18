"""Hooks for revision baa22cad81e2 (standardize_constraint_names)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("baa22cad81e2")
def prepare_standardize_constraint_names(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to baa22cad81e2."""
    # Verify old-style unique constraint names exist
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique"
    )
    verifier.assert_constraint_exists(
        "collections", "unique_observer_observed_collection", "unique"
    )
    verifier.assert_constraint_exists("peers", "unique_name_workspace_peer", "unique")
    verifier.assert_constraint_exists("sessions", "unique_session_name", "unique")
    verifier.assert_constraint_exists("messages", "uq_messages_session_seq", "unique")
    verifier.assert_constraint_exists("workspaces", "uq_apps_name", "unique")

    # Note: FK name checking is flexible because the FK might have been created
    # with different names in different migration paths. The important thing is
    # that after the migration it has the correct name with CASCADE.

    # Verify old-style index names exist
    old_indexes = [
        ("collections", "idx_collections_observed"),
        ("collections", "idx_collections_observer"),
        ("documents", "idx_documents_embedding_hnsw"),
        ("documents", "idx_documents_observed"),
        ("documents", "idx_documents_observer"),
        ("documents", "idx_documents_session_name"),
        ("message_embeddings", "idx_message_embeddings_created_at"),
        ("message_embeddings", "idx_message_embeddings_embedding_hnsw"),
        ("message_embeddings", "idx_message_embeddings_message_id"),
        ("message_embeddings", "idx_message_embeddings_peer_name"),
        ("message_embeddings", "idx_message_embeddings_session_name"),
        ("message_embeddings", "idx_message_embeddings_workspace_name"),
        ("messages", "idx_messages_content_gin"),
        ("messages", "idx_messages_session_lookup"),
        ("webhook_endpoints", "idx_webhook_endpoints_workspace_lookup"),
    ]
    verifier.assert_indexes_exist(old_indexes)

    # Note: We don't check for redundant indexes in before_upgrade because
    # they may or may not exist depending on the migration path. The important
    # thing is that after the migration, they should not exist.


@register_after_upgrade("baa22cad81e2")
def verify_standardize_constraint_names(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of baa22cad81e2."""
    # Verify new-style unique constraint names exist
    verifier.assert_constraint_exists(
        "active_queue_sessions", "uq_active_queue_sessions_work_unit_key", "unique"
    )
    verifier.assert_constraint_exists(
        "collections", "uq_collections_observer_observed_workspace_name", "unique"
    )
    verifier.assert_constraint_exists("peers", "uq_peers_name_workspace_name", "unique")
    verifier.assert_constraint_exists(
        "sessions", "uq_sessions_name_workspace_name", "unique"
    )
    verifier.assert_constraint_exists(
        "messages", "uq_messages_workspace_name_session_name_seq_in_session", "unique"
    )
    verifier.assert_constraint_exists("workspaces", "uq_workspaces_name", "unique")

    # Verify old-style constraint names no longer exist
    verifier.assert_constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique", exists=False
    )
    verifier.assert_constraint_exists(
        "collections", "unique_observer_observed_collection", "unique", exists=False
    )
    verifier.assert_constraint_exists(
        "peers", "unique_name_workspace_peer", "unique", exists=False
    )
    verifier.assert_constraint_exists(
        "sessions", "unique_session_name", "unique", exists=False
    )
    verifier.assert_constraint_exists(
        "messages", "uq_messages_session_seq", "unique", exists=False
    )
    verifier.assert_constraint_exists(
        "workspaces", "uq_apps_name", "unique", exists=False
    )

    # Verify new-style foreign key names exist with correct name
    verifier.assert_constraint_exists(
        "message_embeddings", "fk_message_embeddings_message_id_messages", "foreign_key"
    )

    # Verify new-style index names exist
    new_indexes = [
        ("collections", "ix_collections_observed"),
        ("collections", "ix_collections_observer"),
        ("documents", "ix_documents_embedding_hnsw"),
        ("documents", "ix_documents_observed"),
        ("documents", "ix_documents_observer"),
        ("documents", "ix_documents_session_name"),
        ("message_embeddings", "ix_message_embeddings_created_at"),
        ("message_embeddings", "ix_message_embeddings_embedding_hnsw"),
        ("message_embeddings", "ix_message_embeddings_message_id"),
        ("message_embeddings", "ix_message_embeddings_peer_name"),
        ("message_embeddings", "ix_message_embeddings_session_name"),
        ("message_embeddings", "ix_message_embeddings_workspace_name"),
        ("messages", "ix_messages_content_gin"),
        ("messages", "ix_messages_session_lookup"),
        ("webhook_endpoints", "ix_webhook_endpoints_workspace_name"),
    ]
    verifier.assert_indexes_exist(new_indexes)

    # Verify key old-style index names no longer exist
    # Note: Only checking a subset because some indexes may not exist in all migration paths
    key_old_indexes = [
        ("collections", "idx_collections_observed"),
        ("documents", "idx_documents_observed"),
        ("message_embeddings", "idx_message_embeddings_created_at"),
    ]
    verifier.assert_indexes_not_exist(key_old_indexes)

    # Verify redundant indexes were removed
    redundant_indexes = [
        ("messages", "ix_messages_id"),
        ("messages", "ix_messages_public_id"),
        ("peers", "ix_peers_name"),
        ("peers", "idx_peers_workspace_lookup"),
        ("workspaces", "ix_workspaces_name"),
        ("active_queue_sessions", "ix_active_queue_sessions_work_unit_key"),
        ("queue", "ix_queue_work_unit_key_processed_id"),
        ("queue", "ix_queue_workspace_name_processed"),
    ]
    verifier.assert_indexes_not_exist(redundant_indexes)

    # Verify new indexes were created
    new_queue_indexes = [
        ("queue", "ix_queue_processed"),
        ("queue", "work_unit_key"),
        ("sessions", "ix_sessions_workspace_name"),
    ]
    verifier.assert_indexes_exist(new_queue_indexes)

    # Verify FK was removed
    verifier.assert_constraint_exists(
        "messages", "fk_messages_workspace_name_workspaces", "foreign_key", exists=False
    )

    # Verify FK CASCADE was added to message_embeddings.message_id
    # Check using raw SQL since the verifier doesn't have a method for this
    result = verifier.conn.execute(
        text(
            f"""
                SELECT confdeltype
                FROM pg_constraint
                WHERE conrelid = '{verifier.schema}.message_embeddings'::regclass
                AND conname = 'fk_message_embeddings_message_id_messages'
                """
        )
    )
    row = result.fetchone()
    assert row is not None, "FK fk_message_embeddings_message_id_messages not found"
    assert (
        row[0] == "c"
    ), f"FK should have ON DELETE CASCADE (confdeltype='c'), got '{row[0]}'"
