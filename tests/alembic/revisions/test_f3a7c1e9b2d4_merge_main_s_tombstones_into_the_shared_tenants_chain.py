"""Hooks for revision f3a7c1e9b2d4 (merge_main_s_tombstones_into_the_shared_tenants_chain)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("f3a7c1e9b2d4")
def prepare_merge_main_s_tombstones_into_the_shared_tenants_chain(
    _verifier: MigrationVerifier,
) -> None:
    """A merge revision: no schema change, nothing to seed."""


@register_after_upgrade("f3a7c1e9b2d4")
def verify_merge_main_s_tombstones_into_the_shared_tenants_chain(
    _verifier: MigrationVerifier,
) -> None:
    """A merge revision: no schema change to check."""
