"""Hooks for revision f1a2b3c4d5e6 (reasoning tree columns)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("f1a2b3c4d5e6")
def prepare_reasoning_tree_columns(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("documents", "premise_ids", exists=False)
    verifier.assert_column_exists("documents", "source_ids", exists=False)
    verifier.assert_indexes_not_exist(
        [
            ("documents", "ix_documents_premise_ids_gin"),
            ("documents", "ix_documents_source_ids_gin"),
        ]
    )


@register_after_upgrade("f1a2b3c4d5e6")
def verify_reasoning_tree_columns(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("documents", "premise_ids", nullable=True)
    verifier.assert_column_exists("documents", "source_ids", nullable=True)
    verifier.assert_indexes_exist(
        [
            ("documents", "ix_documents_premise_ids_gin"),
            ("documents", "ix_documents_source_ids_gin"),
        ]
    )
