"""Hooks for revision c3828084f472 (read indexes)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

_INDEXES = (
    ("users", "idx_users_app_lookup"),
    ("sessions", "idx_sessions_user_lookup"),
    ("messages", "idx_messages_session_lookup"),
    ("metamessages", "idx_metamessages_lookup"),
)


@register_before_upgrade("c3828084f472")
def prepare_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(_INDEXES)


@register_after_upgrade("c3828084f472")
def verify_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist(_INDEXES)
