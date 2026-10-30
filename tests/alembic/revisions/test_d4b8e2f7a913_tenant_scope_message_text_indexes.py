"""Hooks for revision d4b8e2f7a913 (tenant-scoped message text-search indexes)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

NEW = [
    ("messages", "ix_messages_tenant_content_fts"),
    ("messages", "ix_messages_tenant_content_trgm"),
]
OLD = [
    ("messages", "ix_messages_content_gin"),
    ("messages", "ix_messages_content_trgm"),
]


@register_before_upgrade("d4b8e2f7a913")
def prepare_tenant_scoped_text_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist(OLD)
    verifier.assert_indexes_not_exist(NEW)


@register_after_upgrade("d4b8e2f7a913")
def verify_tenant_scoped_text_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_valid(NEW)
    verifier.assert_indexes_not_exist(OLD)
