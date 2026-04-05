"""Hooks for revision 7c0d9a4e3b1f (add_unique_index_for_pending_dreams)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# The partial unique index created by this migration
INDEX = ("queue", "uq_queue_dream_pending_work_unit_key")


@register_before_upgrade("7c0d9a4e3b1f")
def prepare_add_unique_index_for_pending_dreams(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to 7c0d9a4e3b1f."""
    # The unique index should not exist before the migration
    verifier.assert_indexes_not_exist([INDEX])


@register_after_upgrade("7c0d9a4e3b1f")
def verify_add_unique_index_for_pending_dreams(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of 7c0d9a4e3b1f."""
    # The partial unique index should now exist
    verifier.assert_indexes_exist([INDEX])
