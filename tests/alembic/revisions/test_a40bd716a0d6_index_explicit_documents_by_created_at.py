"""Hooks for revision a40bd716a0d6 (explicit documents by created_at, for the dream-due poll)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEXES = [("documents", "ix_documents_explicit_created_at")]


@register_before_upgrade("a40bd716a0d6")
def prepare_explicit_created_at_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(INDEXES)


@register_after_upgrade("a40bd716a0d6")
def verify_explicit_created_at_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_valid(INDEXES)
