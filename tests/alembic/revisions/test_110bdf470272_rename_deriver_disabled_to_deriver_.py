"""Hooks for revision 110bdf470272 (rename_deriver_disabled_to_deriver_)."""

from __future__ import annotations

import json

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_ID = generate_nanoid()
WORKSPACE_NAME = "workspace-name"
PEER_ID = generate_nanoid()
PEER_NAME = "peer-name"
SESSION_ID_DISABLED_TRUE = generate_nanoid()
SESSION_ID_DISABLED_FALSE = generate_nanoid()
SESSION_ID_NO_KEY = generate_nanoid()


@register_before_upgrade("110bdf470272")
def prepare_rename_deriver_disabled_to_deriver(verifier: MigrationVerifier) -> None:
    """Seed sessions with deriver_disabled configuration before upgrading to 110bdf470272."""

    schema = verifier.schema
    connection = verifier.conn

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."workspaces" ("id", "name")
                VALUES (:workspace_id, :workspace_name)
                """
        ),
        {"workspace_id": WORKSPACE_ID, "workspace_name": WORKSPACE_NAME},
    )

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
                VALUES (:peer_id, :peer_name, :workspace_name)
                """
        ),
        {
            "peer_id": PEER_ID,
            "peer_name": PEER_NAME,
            "workspace_name": WORKSPACE_NAME,
        },
    )

    configuration_disabled_true = json.dumps({"deriver_disabled": True})
    configuration_disabled_false = json.dumps({"deriver_disabled": False})
    configuration_no_key = json.dumps({"other_setting": "value"})

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."sessions"
                    ("id", "name", "workspace_name", "configuration")
                VALUES (:session_id, :session_name, :workspace_name, :configuration)
                """
        ),
        {
            "session_id": SESSION_ID_DISABLED_TRUE,
            "session_name": "session-disabled-true",
            "workspace_name": WORKSPACE_NAME,
            "configuration": configuration_disabled_true,
        },
    )

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."sessions"
                    ("id", "name", "workspace_name", "configuration")
                VALUES (:session_id, :session_name, :workspace_name, :configuration)
                """
        ),
        {
            "session_id": SESSION_ID_DISABLED_FALSE,
            "session_name": "session-disabled-false",
            "workspace_name": WORKSPACE_NAME,
            "configuration": configuration_disabled_false,
        },
    )

    connection.execute(
        text(
            f"""
                INSERT INTO "{schema}"."sessions"
                    ("id", "name", "workspace_name", "configuration")
                VALUES (:session_id, :session_name, :workspace_name, :configuration)
                """
        ),
        {
            "session_id": SESSION_ID_NO_KEY,
            "session_name": "session-no-key",
            "workspace_name": WORKSPACE_NAME,
            "configuration": configuration_no_key,
        },
    )


@register_after_upgrade("110bdf470272")
def verify_rename_deriver_disabled_to_deriver(verifier: MigrationVerifier) -> None:
    """Verify deriver_disabled was converted to deriver_enabled correctly."""

    schema = verifier.schema
    conn = verifier.conn

    session_disabled_true = conn.execute(
        text(
            f"""
                SELECT configuration FROM "{schema}"."sessions"
                WHERE "id" = :session_id
                """
        ),
        {"session_id": SESSION_ID_DISABLED_TRUE},
    ).one()
    config = session_disabled_true.configuration
    assert "deriver_disabled" not in config, "deriver_disabled should be removed"
    assert (
        config.get("deriver_enabled") is False
    ), "deriver_disabled: true should become deriver_enabled: false"

    session_disabled_false = conn.execute(
        text(
            f"""
                SELECT configuration FROM "{schema}"."sessions"
                WHERE "id" = :session_id
                """
        ),
        {"session_id": SESSION_ID_DISABLED_FALSE},
    ).one()
    config = session_disabled_false.configuration
    assert "deriver_disabled" not in config, "deriver_disabled should be removed"
    assert (
        config.get("deriver_enabled") is True
    ), "deriver_disabled: false should become deriver_enabled: true"

    session_no_key = conn.execute(
        text(
            f"""
                SELECT configuration FROM "{schema}"."sessions"
                WHERE "id" = :session_id
                """
        ),
        {"session_id": SESSION_ID_NO_KEY},
    ).one()
    config = session_no_key.configuration
    assert "deriver_disabled" not in config, "deriver_disabled should not exist"
    assert (
        "deriver_enabled" not in config
    ), "deriver_enabled should not be added when deriver_disabled was absent"
    assert (
        config.get("other_setting") == "value"
    ), "Other configuration should be preserved"
