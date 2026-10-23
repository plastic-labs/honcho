"""Hooks for revision 0c67be41049b (drop_id_only_token_count_fallback)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

ID_ONLY_PROBE = "WHERE id = msg_id"
DEFAULT_TENANT_PROBE = "WHERE tenant_id = 'default' AND id = msg_id"


def _token_count_source(verifier: MigrationVerifier) -> str:
    return verifier.conn.execute(
        text(
            "SELECT pg_get_functiondef(p.oid) FROM pg_proc p "
            + "JOIN pg_namespace n ON n.oid = p.pronamespace "
            + "WHERE p.proname = '_queue_item_token_count' AND n.nspname = :schema"
        ),
        {"schema": verifier.schema},
    ).scalar_one()


@register_before_upgrade("0c67be41049b")
def prepare_drop_id_only_token_count_fallback(verifier: MigrationVerifier) -> None:
    """The token-count function still carries the id-only fallback."""
    source = _token_count_source(verifier)
    assert DEFAULT_TENANT_PROBE in source
    assert ID_ONLY_PROBE in source


@register_after_upgrade("0c67be41049b")
def verify_drop_id_only_token_count_fallback(verifier: MigrationVerifier) -> None:
    """Only the default-tenant probe remains for tenant-less rows."""
    source = _token_count_source(verifier)
    assert DEFAULT_TENANT_PROBE in source
    assert ID_ONLY_PROBE not in source
