"""Hooks for revision b7d2e4f19c03 (add_collection_explicit_counters)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = generate_nanoid()
PEER_A = generate_nanoid()
PEER_B = generate_nanoid()


@register_before_upgrade("b7d2e4f19c03")
def prepare_add_collection_explicit_counters(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("collections", "explicit_count", exists=False)
    schema = verifier.schema
    conn = verifier.conn
    conn.execute(
        text(
            f'INSERT INTO "{schema}"."workspaces" (tenant_id, id, name)'
            + " VALUES ('default', :id, :name)"
        ),
        {"id": generate_nanoid(), "name": WORKSPACE_NAME},
    )
    for peer in (PEER_A, PEER_B):
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."peers" (tenant_id, id, name, workspace_name)'
                + " VALUES ('default', :id, :name, :ws)"
            ),
            {"id": generate_nanoid(), "name": peer, "ws": WORKSPACE_NAME},
        )
    for observed in (PEER_A, PEER_B):
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."collections"'
                + " (tenant_id, id, workspace_name, observer, observed)"
                + " VALUES ('default', :id, :ws, :observer, :observed)"
            ),
            {
                "id": generate_nanoid(),
                "ws": WORKSPACE_NAME,
                "observer": PEER_A,
                "observed": observed,
            },
        )
    for level, deleted, created in (
        ("explicit", False, "2026-01-01T00:00:00Z"),
        ("explicit", True, "2026-02-01T00:00:00Z"),
        ("deductive", False, "2026-03-01T00:00:00Z"),
    ):
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."documents"'
                + " (tenant_id, id, workspace_name, observer, observed, content, level,"
                + " created_at, deleted_at)"
                + " VALUES ('default', :id, :ws, :observer, :observed, 'c', :level,"
                + " CAST(:created AS timestamptz),"
                + " CASE WHEN :deleted THEN now() END)"
            ),
            {
                "id": generate_nanoid(),
                "ws": WORKSPACE_NAME,
                "observer": PEER_A,
                "observed": PEER_A,
                "level": level,
                "created": created,
                "deleted": deleted,
            },
        )


@register_after_upgrade("b7d2e4f19c03")
def verify_add_collection_explicit_counters(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("collections", "explicit_count", nullable=False)
    verifier.assert_column_exists("collections", "newest_explicit_at", nullable=True)
    verifier.assert_indexes_exist(
        [
            ("collections", "ix_collections_dream_due"),
            ("collections", "ix_collections_dream_backlog"),
        ]
    )
    rows = verifier.conn.execute(
        text(
            "SELECT observed, explicit_count, newest_explicit_at::text"
            + f' FROM "{verifier.schema}"."collections"'
            + " WHERE workspace_name = :ws"
        ),
        {"ws": WORKSPACE_NAME},
    ).all()
    counters = {row[0]: (row[1], row[2]) for row in rows}
    assert counters[PEER_A][0] == 2
    assert counters[PEER_A][1] is not None
    assert counters[PEER_A][1].startswith("2026-02-01")
    assert counters[PEER_B] == (0, None)
