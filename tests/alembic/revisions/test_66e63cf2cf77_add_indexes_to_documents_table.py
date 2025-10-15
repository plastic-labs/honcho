"""Hooks for revision 66e63cf2cf77 (documents HNSW index)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("66e63cf2cf77")
def verify_documents_hnsw(verifier: MigrationVerifier) -> None:
    """Ensure the HNSW index on documents.embedding exists."""

    verifier.assert_index_exists("documents", "idx_documents_embedding_hnsw")
