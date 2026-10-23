"""Preserve installed progress and tombstone branches through their merge."""

from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from nanoid import generate as generate_nanoid
from sqlalchemy import Engine, text

from src.config import settings
from tests.alembic.revisions.test_c6d9e2f74a10_merge_progress_and_pending_work import (
    PROGRESS_COLUMNS,
)
from tests.alembic.revisions.test_f2a6d9c81b04_merge_progress_and_primitive_tombstones import (
    REVISION,
    TOMBSTONE_TABLES,
    verify_progress_and_tombstone_schema,
)
from tests.alembic.verifier import MigrationVerifier

COMMON_PARENT = "a5f8da8f6d23"
PROGRESS_PARENT = "c6d9e2f74a10"
TOMBSTONE_PARENT = "d8c4f6a2b901"
PROGRESS_DDL_REVISION = "a7c2e4f819d0"


def _versions(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(conn.scalars(text("SELECT version_num FROM alembic_version")))


def _claims(engine: Engine) -> list[tuple[object, ...]]:
    with engine.connect() as conn:
        return [
            tuple(row)
            for row in conn.execute(
                text(
                    "SELECT id, started_at, last_activity_at, stage, current_queue_item_id"
                    + f' FROM "{settings.DB.SCHEMA}".active_queue_sessions ORDER BY id'
                )
            )
        ]


def _assert_primitive_rows(engine: Engine, tombstone: datetime | None) -> None:
    schema = settings.DB.SCHEMA
    with engine.connect() as conn:
        for table in TOMBSTONE_TABLES:
            workspace_column = "name" if table == "workspaces" else "workspace_name"
            assert conn.execute(
                text(
                    f'SELECT deleted_at FROM "{schema}".{table}'
                    + f" WHERE {workspace_column} = 'merge-primitives'"
                )
            ).one() == (tombstone,)
        assert conn.execute(
            text(
                f'SELECT (SELECT deletion_generation FROM "{schema}".workspaces'
                + " WHERE name = 'merge-primitives'),"
                + f' (SELECT deletion_generation FROM "{schema}".sessions'
                + " WHERE workspace_name = 'merge-primitives'),"
                + f' (SELECT content FROM "{schema}".messages'
                + " WHERE workspace_name = 'merge-primitives')"
            )
        ).one() == (7, 11, "legacy content")


def test_fresh_upgrade_includes_progress_and_tombstones(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, REVISION)
    assert _versions(alembic_engine) == {REVISION}
    with alembic_engine.connect() as conn:
        verify_progress_and_tombstone_schema(MigrationVerifier(conn, REVISION))


@pytest.mark.parametrize(
    "starting_revision", [COMMON_PARENT, PROGRESS_PARENT, TOMBSTONE_PARENT, "both"]
)
def test_tombstone_progress_upgrade_and_downgrade(
    alembic_cfg: Config, alembic_engine: Engine, starting_revision: str
) -> None:
    if starting_revision == "both":
        command.upgrade(alembic_cfg, PROGRESS_PARENT)
        command.upgrade(alembic_cfg, TOMBSTONE_PARENT)
        assert _versions(alembic_engine) == {PROGRESS_PARENT, TOMBSTONE_PARENT}
    else:
        command.upgrade(alembic_cfg, starting_revision)
        assert _versions(alembic_engine) == {starting_revision}

    schema = settings.DB.SCHEMA
    has_progress = starting_revision in (PROGRESS_PARENT, "both")
    has_tombstones = starting_revision in (TOMBSTONE_PARENT, "both")
    stamp = datetime(2026, 10, 23, tzinfo=UTC)
    with alembic_engine.begin() as conn:
        for table, columns, values in (
            (
                "workspaces",
                "id, name, deletion_generation",
                ":uid, 'merge-primitives', 7",
            ),
            (
                "peers",
                "id, name, workspace_name",
                ":uid, 'author', 'merge-primitives'",
            ),
            (
                "sessions",
                "id, name, workspace_name, deletion_generation",
                ":uid, 'session', 'merge-primitives', 11",
            ),
            (
                "messages",
                "public_id, workspace_name, session_name, peer_name, content, token_count, seq_in_session",
                ":uid, 'merge-primitives', 'session', 'author', 'legacy content', 2, 1",
            ),
        ):
            conn.execute(
                text(f'INSERT INTO "{schema}".{table} ({columns}) VALUES ({values})'),
                {"uid": generate_nanoid()},
            )
        conn.execute(
            text(
                f'INSERT INTO "{schema}".active_queue_sessions (id, work_unit_key)'
                + " VALUES ('tombstone-claim', 'tombstone-claim')"
            )
        )
        if has_progress:
            conn.execute(
                text(
                    f'UPDATE "{schema}".active_queue_sessions SET'
                    + " started_at = :stamp, last_activity_at = :stamp,"
                    + " stage = 'deduction', current_queue_item_id = 123"
                ),
                {"stamp": stamp},
            )
        if has_tombstones:
            for table in TOMBSTONE_TABLES:
                conn.execute(
                    text(f'UPDATE "{schema}".{table} SET deleted_at = :stamp'),
                    {"stamp": stamp},
                )

    command.upgrade(alembic_cfg, REVISION)
    assert _versions(alembic_engine) == {REVISION}
    _assert_primitive_rows(alembic_engine, stamp if has_tombstones else None)
    with alembic_engine.begin() as conn:
        verify_progress_and_tombstone_schema(MigrationVerifier(conn, REVISION))
        claim = conn.execute(
            text(
                "SELECT started_at, last_activity_at, stage, current_queue_item_id"
                + f' FROM "{schema}".active_queue_sessions'
            )
        ).one()
        assert tuple(claim) == (
            (stamp, stamp, "deduction", 123) if has_progress else (None,) * 4
        )
        fresh = conn.execute(
            text(
                f'INSERT INTO "{schema}".active_queue_sessions (id, work_unit_key)'
                + " VALUES ('fresh-claim', 'fresh-claim')"
                + " RETURNING started_at, last_activity_at, stage"
            )
        ).one()
        assert (
            fresh.started_at is not None and fresh.started_at == fresh.last_activity_at
        )
        assert fresh.stage == "claimed"

    observed_claims = _claims(alembic_engine)
    command.downgrade(alembic_cfg, COMMON_PARENT)
    # a7 is a sibling of a5, so this branch rollback deliberately leaves it installed.
    assert _versions(alembic_engine) == {COMMON_PARENT, PROGRESS_DDL_REVISION}
    assert _claims(alembic_engine) == observed_claims
    with alembic_engine.connect() as conn:
        verifier = MigrationVerifier(conn, COMMON_PARENT)
        for column in PROGRESS_COLUMNS:
            verifier.assert_column_exists(
                "active_queue_sessions", column, nullable=True
            )
        for table in TOMBSTONE_TABLES:
            verifier.assert_column_exists(table, "deleted_at", exists=False)
        assert (
            conn.scalar(text(f'SELECT count(*) FROM "{schema}".active_queue_sessions'))
            == 2
        )

    # Only the tombstone columns were removed; progress and generation values survive.
    command.upgrade(alembic_cfg, REVISION)
    assert _versions(alembic_engine) == {REVISION}
    _assert_primitive_rows(alembic_engine, None)
    assert _claims(alembic_engine) == observed_claims
    with alembic_engine.connect() as conn:
        verify_progress_and_tombstone_schema(MigrationVerifier(conn, REVISION))
