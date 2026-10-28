"""Hooks for revision a1e7ce5396d5 (message_text_matches_function)."""

from __future__ import annotations

from sqlalchemy import text

from src.models import MESSAGE_TEXT_MATCHES
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


def _function_row(verifier: MigrationVerifier) -> tuple[bool, str] | None:
    row = verifier.conn.execute(
        text(
            "SELECT p.prosecdef, p.proconfig::text FROM pg_proc p"
            + " JOIN pg_namespace n ON n.oid = p.pronamespace"
            + " WHERE n.nspname = :schema AND p.proname = :name"
        ),
        {"schema": verifier.schema, "name": MESSAGE_TEXT_MATCHES},
    ).first()
    return None if row is None else (row[0], row[1])


@register_before_upgrade("a1e7ce5396d5")
def prepare_message_text_matches_function(verifier: MigrationVerifier) -> None:
    assert _function_row(verifier) is None


@register_after_upgrade("a1e7ce5396d5")
def verify_message_text_matches_function(verifier: MigrationVerifier) -> None:
    row = _function_row(verifier)
    assert row is not None
    security_definer, config = row
    assert security_definer is True
    assert "search_path=pg_catalog, pg_temp" in config

    schema = verifier.schema
    conn = verifier.conn
    conn.execute(text("SELECT set_config('app.tenant', 'default', false)"))
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" (tenant_id, id, name)'
            + " VALUES ('default', 'a1e7ce5396d5a1e7ce539', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."peers" (tenant_id, id, name, workspace_name)'
            + " VALUES ('default', 'a1e7ce5396d5a1e7ce53a', 'p', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."sessions" (tenant_id, id, name, workspace_name)'
            + " VALUES ('default', 'a1e7ce5396d5a1e7ce53b', 's', 'w')"
        )
    )
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."messages"'
            + " (tenant_id, id, public_id, workspace_name, session_name, peer_name,"
            + " content, seq_in_session, token_count)"
            + " VALUES ('default', 1, 'a1e7ce5396d5a1e7ce53c', 'w', 's', 'p',"
            + " 'the quick brown fox', 1, 4),"
            + " ('default', 2, 'a1e7ce5396d5a1e7ce53d', 'w', 's', 'p',"
            + " 'nothing here', 2, 2)"
        )
    )
    matched = conn.execute(
        text(
            f'SELECT * FROM "{schema}".{MESSAGE_TEXT_MATCHES}'
            + "('default', 'w', 'fox', '%fox%', true)"
        )
    ).scalars()
    assert list(matched) == [1]
    unbound = conn.execute(
        text(
            f'SELECT * FROM "{schema}".{MESSAGE_TEXT_MATCHES}'
            + "('other', 'w', 'fox', '%fox%', true)"
        )
    ).scalars()
    assert list(unbound) == []
