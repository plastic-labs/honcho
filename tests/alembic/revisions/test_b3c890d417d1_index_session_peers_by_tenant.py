"""Hooks for revision b3c890d417d1 (session_peers index with tenant_id first)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

NEW = [("session_peers", "ix_session_peers_tenant_workspace_peer")]
OLD = [("session_peers", "ix_session_peers_workspace_peer")]


@register_before_upgrade("b3c890d417d1")
def prepare_session_peers_tenant_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(NEW)
    verifier.assert_indexes_valid(OLD)


@register_after_upgrade("b3c890d417d1")
def verify_session_peers_tenant_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_valid(NEW)
    verifier.assert_indexes_not_exist(OLD)
