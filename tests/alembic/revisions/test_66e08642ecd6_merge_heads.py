"""Hooks for revision 66e08642ecd6 (merge_heads)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("66e08642ecd6")
def prepare_merge_heads(_verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to 66e08642ecd6."""


@register_after_upgrade("66e08642ecd6")
def verify_merge_heads(_verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of 66e08642ecd6."""
