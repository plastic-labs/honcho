"""Hooks for revision f2a70ec72da0 (peer_memberships table)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

TABLE = "peer_memberships"
INDEXES = [
    (TABLE, "ix_peer_memberships_parent"),
    (TABLE, "ix_peer_memberships_member"),
]


@register_before_upgrade("f2a70ec72da0")
def prepare_peer_memberships(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists(TABLE, exists=False)


@register_after_upgrade("f2a70ec72da0")
def verify_peer_memberships(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists(TABLE)
    verifier.assert_indexes_exist(INDEXES)
    verifier.assert_constraint_exists(
        TABLE, "uq_peer_memberships_workspace_name_parent_id_member_id", "unique"
    )
    verifier.assert_constraint_exists(TABLE, "ck_peer_memberships_not_self", "check")
