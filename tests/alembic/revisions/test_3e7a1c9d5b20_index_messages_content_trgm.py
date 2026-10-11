"""Hooks for revision 3e7a1c9d5b20 (messages.content trigram index)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEX = ("messages", "ix_messages_content_trgm")


@register_before_upgrade("3e7a1c9d5b20")
def prepare_messages_content_trgm_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist([INDEX])


@register_after_upgrade("3e7a1c9d5b20")
def verify_messages_content_trgm_index(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist([INDEX])
