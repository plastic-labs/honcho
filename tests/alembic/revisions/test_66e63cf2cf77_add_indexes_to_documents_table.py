"""Hooks for revision 66e63cf2cf77 (documents HNSW index)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("66e63cf2cf77")
def prepare_documents_hnsw(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist([("documents", "idx_documents_embedding_hnsw")])


@register_after_upgrade("66e63cf2cf77")
def verify_documents_hnsw(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist([("documents", "idx_documents_embedding_hnsw")])
