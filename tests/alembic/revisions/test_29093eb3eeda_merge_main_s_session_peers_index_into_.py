"""Hooks for revision 29093eb3eeda (merge_main_s_session_peers_index_into_)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("29093eb3eeda")
def prepare_merge_main_s_session_peers_index_into(_verifier: MigrationVerifier) -> None:
    """Merge revision with no schema change; nothing to seed."""


@register_after_upgrade("29093eb3eeda")
def verify_merge_main_s_session_peers_index_into(_verifier: MigrationVerifier) -> None:
    """Merge revision with no schema change; nothing to verify."""
