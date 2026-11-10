"""Hooks for revision a7e3c1d9f2b4 (dream history tables, DDL only)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = "dream_history_ws"
RUN_ID = generate_nanoid()


@register_before_upgrade("a7e3c1d9f2b4")
def prepare_dream_history(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("dream_runs", exists=False)
    verifier.assert_table_exists("dream_run_changes", exists=False)
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}"."workspaces" (tenant_id, id, name)'
            + " VALUES ('default', :id, :n)"
        ),
        {"id": generate_nanoid(), "n": WORKSPACE_NAME},
    )


@register_after_upgrade("a7e3c1d9f2b4")
def verify_dream_history(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("dream_runs")
    verifier.assert_table_exists("dream_run_changes")
    verifier.assert_column_exists("dream_runs", "pruned_at")
    verifier.assert_indexes_exist(
        [
            ("dream_runs", "ix_dream_runs_tenant_workspace_observed_started"),
            ("dream_runs", "ix_dream_runs_started_at"),
            ("dream_run_changes", "ix_dream_run_changes_tenant_run_id"),
            ("dream_run_changes", "ix_dream_run_changes_removed_conclusion"),
            ("dream_run_changes", "ix_dream_run_changes_session"),
        ]
    )

    schema = verifier.schema
    connection = verifier.conn

    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."dream_runs"
                ("tenant_id", "id", "workspace_name", "observer", "observed")
            VALUES ('default', :id, :w, 'alice', 'alice')"""
        ),
        {"id": RUN_ID, "w": WORKSPACE_NAME},
    )
    status = connection.execute(
        text(f'SELECT "status" FROM "{schema}"."dream_runs" WHERE "id" = :id'),
        {"id": RUN_ID},
    ).scalar_one()
    assert status == "running"

    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."dream_run_changes"
                ("tenant_id", "run_id", "workspace_name", "kind",
                 "conclusion_id", "content")
            VALUES ('default', :r, :w, 'removed', :c, 'an old fact')"""
        ),
        {"r": RUN_ID, "w": WORKSPACE_NAME, "c": generate_nanoid()},
    )

    # Changes go with their run.
    connection.execute(
        text(f'DELETE FROM "{schema}"."dream_runs" WHERE "id" = :id'), {"id": RUN_ID}
    )
    change_count = connection.execute(
        text(f'SELECT count(*) FROM "{schema}"."dream_run_changes"')
    ).scalar_one()
    assert change_count == 0
