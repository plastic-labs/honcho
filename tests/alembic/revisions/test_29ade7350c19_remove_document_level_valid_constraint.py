"""Hooks for revision 29ade7350c19 (remove_document_level_valid_constraint)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("29ade7350c19")
def prepare_remove_document_level_valid_constraint(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to 29ade7350c19."""
    # Verify CHECK constraint exists before migration (from b8183c5ffb48)
    verifier.assert_constraint_exists("documents", "level_valid", "check")


@register_after_upgrade("29ade7350c19")
def verify_remove_document_level_valid_constraint(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of 29ade7350c19."""
    # Verify CHECK constraint no longer exists after migration
    verifier.assert_constraint_exists("documents", "level_valid", "check", exists=False)
