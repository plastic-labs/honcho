"""Hooks for revision b8d2f4a6c9e1 (session_peers peer-lookup index)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEX = ("session_peers", "ix_session_peers_workspace_peer")


@register_before_upgrade("b8d2f4a6c9e1")
def prepare_session_peers_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist([INDEX])


@register_after_upgrade("b8d2f4a6c9e1")
def verify_session_peers_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist([INDEX])
