"""Hooks for revision a8f2d4e6c9b1 (add_dialectic_traces_table)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEXES = (
    ("dialectic_traces", "ix_dialectic_traces_workspace_name"),
    ("dialectic_traces", "ix_dialectic_traces_session_name"),
    ("dialectic_traces", "ix_dialectic_traces_observer"),
    ("dialectic_traces", "ix_dialectic_traces_observed"),
    ("dialectic_traces", "ix_dialectic_traces_created_at"),
)


@register_before_upgrade("a8f2d4e6c9b1")
def prepare_dialectic_traces(verifier: MigrationVerifier) -> None:
    """Assert dialectic_traces table does not exist before migration."""
    verifier.assert_table_exists("dialectic_traces", exists=False)


@register_after_upgrade("a8f2d4e6c9b1")
def verify_dialectic_traces_table(verifier: MigrationVerifier) -> None:
    """Validate dialectic_traces table and indexes after migration."""
    verifier.assert_table_exists("dialectic_traces")

    # Verify columns exist with correct nullability
    verifier.assert_column_exists("dialectic_traces", "id", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "workspace_name", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "session_name", nullable=True)
    verifier.assert_column_exists("dialectic_traces", "observer", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "observed", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "query", nullable=False)
    verifier.assert_column_exists(
        "dialectic_traces", "retrieved_doc_ids", nullable=False
    )
    verifier.assert_column_exists("dialectic_traces", "tool_calls", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "response", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "reasoning_level", nullable=False)
    verifier.assert_column_exists(
        "dialectic_traces", "total_duration_ms", nullable=False
    )
    verifier.assert_column_exists("dialectic_traces", "input_tokens", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "output_tokens", nullable=False)
    verifier.assert_column_exists("dialectic_traces", "created_at", nullable=False)

    # Verify indexes
    verifier.assert_indexes_exist(INDEXES)

    # Verify foreign key constraint
    verifier.assert_constraint_exists(
        "dialectic_traces", "fk_dialectic_traces_workspace_name", "foreign_key"
    )
