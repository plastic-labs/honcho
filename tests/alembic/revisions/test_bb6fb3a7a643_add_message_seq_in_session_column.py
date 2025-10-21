"""Hooks for revision bb6fb3a7a643 (add_message_seq_in_session_column)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("bb6fb3a7a643")
def prepare_add_message_seq_in_session_column(_verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to bb6fb3a7a643."""


@register_after_upgrade("bb6fb3a7a643")
def verify_add_message_seq_in_session_column(_verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of bb6fb3a7a643."""
