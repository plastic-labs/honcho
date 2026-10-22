"""Hooks for revision c075274557cd (merge_main_s_messages_trigram_index_into_)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("c075274557cd")
def prepare_merge_main_s_messages_trigram_index_into(
    _verifier: MigrationVerifier,
) -> None:
    """A merge revision: no schema change, nothing to seed."""


@register_after_upgrade("c075274557cd")
def verify_merge_main_s_messages_trigram_index_into(
    _verifier: MigrationVerifier,
) -> None:
    """A merge revision: no schema change to check."""
