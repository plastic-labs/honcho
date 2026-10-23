"""Upgrade either published branch, both branches, or their common ancestor."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings
from tests.alembic.revisions.test_c6d9e2f74a10_merge_progress_and_pending_work import (
    GENERATION_TABLES,
    PROGRESS_COLUMNS,
    REVISION,
    verify_merged_schema,
)
from tests.alembic.verifier import MigrationVerifier

COMMON_PARENT = "3e7a1c9d5b20"
PROGRESS_PARENT = "a7c2e4f819d0"
PENDING_PARENT = "a5f8da8f6d23"


def _versions(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(conn.scalars(text("SELECT version_num FROM alembic_version")))


@pytest.mark.parametrize(
    "starting_revision", [COMMON_PARENT, PROGRESS_PARENT, PENDING_PARENT, "both"]
)
def test_progress_merge_upgrade_and_downgrade(
    alembic_cfg: Config, alembic_engine: Engine, starting_revision: str
) -> None:
    if starting_revision == "both":
        command.upgrade(alembic_cfg, PENDING_PARENT)
        command.upgrade(alembic_cfg, PROGRESS_PARENT)
        assert _versions(alembic_engine) == {PENDING_PARENT, PROGRESS_PARENT}
    else:
        command.upgrade(alembic_cfg, starting_revision)
        assert _versions(alembic_engine) == {starting_revision}

    schema = settings.DB.SCHEMA
    has_progress = starting_revision in (PROGRESS_PARENT, "both")
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO "{schema}".active_queue_sessions (id, work_unit_key)'
                + " VALUES ('merge-claim', 'merge-claim')"
            )
        )
        if has_progress:
            conn.execute(
                text(
                    f'UPDATE "{schema}".active_queue_sessions SET'
                    + " started_at = '2026-10-20T01:00:00Z',"
                    + " last_activity_at = '2026-10-20T01:01:00Z',"
                    + " stage = 'deduction', current_queue_item_id = 123"
                    + " WHERE id = 'merge-claim'"
                )
            )
            observed = tuple(
                conn.execute(
                    text(
                        "SELECT started_at, last_activity_at, stage, current_queue_item_id"
                        + f' FROM "{schema}".active_queue_sessions'
                        + " WHERE id = 'merge-claim'"
                    )
                ).one()
            )
        else:
            observed = (None, None, None, None)

    # Use env.py's owning-transaction path, including concurrent index builds.
    command.upgrade(alembic_cfg, REVISION)
    assert _versions(alembic_engine) == {REVISION}
    with alembic_engine.begin() as conn:
        verify_merged_schema(MigrationVerifier(conn, REVISION))
        claim = conn.execute(
            text(
                "SELECT started_at, last_activity_at, stage, current_queue_item_id"
                + f' FROM "{schema}".active_queue_sessions'
                + " WHERE id = 'merge-claim'"
            )
        ).one()
        assert tuple(claim) == observed
        fresh = conn.execute(
            text(
                f'INSERT INTO "{schema}".active_queue_sessions (id, work_unit_key)'
                + " VALUES ('merge-new', 'merge-new')"
                + " RETURNING started_at, last_activity_at, stage"
            )
        ).one()
        assert fresh.started_at is not None
        assert fresh.started_at == fresh.last_activity_at
        assert fresh.stage == "claimed"

    command.downgrade(alembic_cfg, COMMON_PARENT)
    assert _versions(alembic_engine) == {COMMON_PARENT}
    with alembic_engine.connect() as conn:
        verifier = MigrationVerifier(conn, COMMON_PARENT)
        for column in PROGRESS_COLUMNS:
            verifier.assert_column_exists("active_queue_sessions", column, exists=False)
        for table in GENERATION_TABLES:
            verifier.assert_column_exists(table, "deletion_generation", exists=False)
        assert (
            conn.scalar(
                text(
                    f'SELECT count(*) FROM "{schema}".active_queue_sessions'
                    + " WHERE id IN ('merge-claim', 'merge-new')"
                )
            )
            == 2
        )

    command.upgrade(alembic_cfg, REVISION)
    assert _versions(alembic_engine) == {REVISION}
    with alembic_engine.connect() as conn:
        verify_merged_schema(MigrationVerifier(conn, REVISION))
        rows = conn.execute(
            text(
                "SELECT started_at, last_activity_at, stage, current_queue_item_id"
                + f' FROM "{schema}".active_queue_sessions ORDER BY id'
            )
        ).all()
        assert [tuple(row) for row in rows] == [(None, None, None, None)] * 2
