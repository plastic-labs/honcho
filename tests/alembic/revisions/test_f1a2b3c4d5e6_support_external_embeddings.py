"""Hooks for revision f1a2b3c4d5e6 (support_external_embeddings)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("f1a2b3c4d5e6")
def prepare_support_external_embeddings(
    verifier: MigrationVerifier,
) -> None:
    """Seed state and assertions before upgrading to f1a2b3c4d5e6."""
    verifier.assert_column_exists("message_embeddings", "embedding", nullable=False)


@register_after_upgrade("f1a2b3c4d5e6")
def verify_support_external_embeddings(
    verifier: MigrationVerifier,
) -> None:
    """Add assertions validating the effects of f1a2b3c4d5e6."""
    verifier.assert_column_exists("message_embeddings", "embedding", nullable=True)
