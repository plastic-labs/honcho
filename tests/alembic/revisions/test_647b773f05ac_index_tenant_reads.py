"""Hooks for revision 647b773f05ac (indexes for the reads that grow with one tenant's data)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEXES = [
    ("documents", "ix_documents_workspace_recent"),
    ("documents", "ix_documents_collection_derived"),
    ("messages", "ix_messages_session_tokens"),
]


@register_before_upgrade("647b773f05ac")
def prepare_tenant_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(INDEXES)


@register_after_upgrade("647b773f05ac")
def verify_tenant_read_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_valid(INDEXES)
