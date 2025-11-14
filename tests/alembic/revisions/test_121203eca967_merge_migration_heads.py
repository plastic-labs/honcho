"""Hooks for revision 121203eca967 (merge_migration_heads)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("121203eca967")
def prepare_merge_migration_heads(_verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to 121203eca967."""


@register_after_upgrade("121203eca967")
def verify_merge_migration_heads(_verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of 121203eca967."""
