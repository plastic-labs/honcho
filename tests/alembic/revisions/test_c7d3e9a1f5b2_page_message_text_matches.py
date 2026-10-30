"""Hooks for revision c7d3e9a1f5b2 (page_message_text_matches)."""

from __future__ import annotations

from sqlalchemy import text

from src.models import MESSAGE_TEXT_MATCHES
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

LEGACY = "message_text_matches"


def _function(verifier: MigrationVerifier, name: str) -> tuple[bool, bool] | None:
    """(prosecdef, PUBLIC may execute) for the function, or None if absent."""
    row = verifier.conn.execute(
        text(
            "SELECT p.prosecdef, has_function_privilege('public', p.oid, 'EXECUTE')"
            + " FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace"
            + " WHERE n.nspname = :schema AND p.proname = :name"
        ),
        {"schema": verifier.schema, "name": name},
    ).first()
    return None if row is None else (row[0], row[1])


@register_before_upgrade("c7d3e9a1f5b2")
def prepare_page_message_text_matches(verifier: MigrationVerifier) -> None:
    assert _function(verifier, MESSAGE_TEXT_MATCHES) is None
    assert _function(verifier, LEGACY) == (True, True)


@register_after_upgrade("c7d3e9a1f5b2")
def verify_page_message_text_matches(verifier: MigrationVerifier) -> None:
    assert _function(verifier, MESSAGE_TEXT_MATCHES) == (True, False)
    assert _function(verifier, LEGACY) == (True, False)

    schema = verifier.schema
    conn = verifier.conn
    conn.execute(text("SELECT set_config('app.tenant', 'default', false)"))
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" (tenant_id, id, name)'
            + " VALUES ('default', 'c7d3e9a1f5b2c7d3e9a1f', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" (tenant_id, id, name, workspace_name)'
            + " VALUES ('default', 'c7d3e9a1f5b2c7d3e9a1a', 'p', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" (tenant_id, id, name, workspace_name)'
            + " VALUES ('default', 'c7d3e9a1f5b2c7d3e9a1b', 's', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."messages"'
            + " (tenant_id, id, public_id, workspace_name, session_name, peer_name,"
            + " content, seq_in_session, token_count, created_at)"
            + " VALUES ('default', 1, 'c7d3e9a1f5b2c7d3e9a1c', 'w', 's', 'p',"
            + " 'an old fox', 1, 3, '2026-01-01T00:00:00Z'),"
            + " ('default', 2, 'c7d3e9a1f5b2c7d3e9a1d', 'w', 's', 'p',"
            + " 'a new fox', 2, 3, '2026-02-01T00:00:00Z'),"
            + " ('default', 3, 'c7d3e9a1f5b2c7d3e9a1e', 'w', 's', 'p',"
            + " 'nothing here', 3, 2, '2026-03-01T00:00:00Z')"
        )
    )
    page = conn.execute(
        text(
            f'SELECT id FROM "{schema}".{MESSAGE_TEXT_MATCHES}'
            + "('default', 'w', 'fox', '%fox%', false, NULL, NULL, NULL, 1)"
        )
    ).scalars()
    assert list(page) == [2]
