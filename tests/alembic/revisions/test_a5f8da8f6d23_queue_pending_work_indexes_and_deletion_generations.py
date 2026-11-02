"""Hooks for revision a5f8da8f6d23 (queue pending-work indexes, deletion generations)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

# Generation table -> its workspace column
GENERATION_TABLES = {
    "workspaces": "name",
    "sessions": "workspace_name",
    "collections": "workspace_name",
}
INDEXES = [
    ("queue", "ix_queue_pending_observers"),
    ("queue", "ix_queue_pending_observed"),
    ("queue", "ix_queue_pending_observer"),
    ("queue", "ix_queue_pending_scope_peer"),
    ("queue", "ix_queue_pending_session_name"),
]
WORKSPACE = "ws_generation"


@register_before_upgrade("a5f8da8f6d23")
def prepare_pending_work_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_not_exist(INDEXES)
    for table in GENERATION_TABLES:
        verifier.assert_column_exists(table, "deletion_generation", exists=False)

    schema = verifier.schema
    verifier.conn.execute(
        text(
            f'INSERT INTO "{schema}".workspaces (id, name)'
            + " VALUES ('wsgen0000000000000000', :ws)"
        ),
        {"ws": WORKSPACE},
    )
    verifier.conn.execute(
        text(
            f'INSERT INTO "{schema}".peers (id, name, workspace_name)'
            + " VALUES ('peergen00000000000000', 'alice', :ws)"
        ),
        {"ws": WORKSPACE},
    )
    verifier.conn.execute(
        text(
            f'INSERT INTO "{schema}".sessions (id, name, workspace_name)'
            + " VALUES ('sessgen00000000000000', 's1', :ws)"
        ),
        {"ws": WORKSPACE},
    )
    verifier.conn.execute(
        text(
            f'INSERT INTO "{schema}".collections (id, observer, observed, workspace_name)'
            + " VALUES ('collgen00000000000000', 'alice', 'alice', :ws)"
        ),
        {"ws": WORKSPACE},
    )


@register_after_upgrade("a5f8da8f6d23")
def verify_pending_work_indexes(verifier: MigrationVerifier) -> None:
    verifier.assert_indexes_exist(INDEXES)
    for table, workspace_column in GENERATION_TABLES.items():
        verifier.assert_column_exists(
            table, "deletion_generation", exists=True, nullable=False
        )
        verifier.assert_no_nulls(table, "deletion_generation")
        generations = verifier.conn.scalars(
            text(
                f'SELECT deletion_generation FROM "{verifier.schema}".{table}'
                + f" WHERE {workspace_column} = :ws"
            ),
            {"ws": WORKSPACE},
        ).all()
        assert generations == [0]
