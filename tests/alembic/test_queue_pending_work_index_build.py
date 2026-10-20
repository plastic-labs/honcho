"""The concurrent path of revision a5f8da8f6d23 (queue pending-work indexes).

test_pipeline.py runs revisions on a caller-owned connection, which takes the
transactional CREATE INDEX branch and never downgrades. These run
``command.upgrade`` / ``command.downgrade`` the way ``src.migrate`` does, so
env.py owns the transaction and the indexes are built CONCURRENTLY. The
invalid-leftover case also runs on a caller-owned connection.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings

REVISION = "a5f8da8f6d23"
DOWN_REVISION = "3e7a1c9d5b20"
INDEXES = (
    "ix_queue_pending_observers",
    "ix_queue_pending_observed",
    "ix_queue_pending_observer",
    "ix_queue_pending_scope_peer",
    "ix_queue_pending_session_name",
)
GENERATION_TABLES = ("workspaces", "sessions", "collections")


def _index_states(alembic_engine: Engine) -> dict[str, tuple[int, bool]]:
    """Return {index name: (oid, indisvalid)} for the revision's indexes."""
    with alembic_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT c.relname, i.indexrelid::bigint, i.indisvalid FROM pg_index i"
                + " JOIN pg_class c ON c.oid = i.indexrelid"
                + " JOIN pg_namespace n ON n.oid = c.relnamespace"
                + " WHERE c.relname = ANY(:names) AND n.nspname = :schema"
            ),
            {"names": list(INDEXES), "schema": settings.DB.SCHEMA},
        ).all()
    return {row[0]: (row[1], row[2]) for row in rows}


def _generation_tables(alembic_engine: Engine) -> set[str]:
    with alembic_engine.connect() as conn:
        return set(
            conn.scalars(
                text(
                    "SELECT table_name FROM information_schema.columns"
                    + " WHERE table_schema = :schema"
                    + " AND column_name = 'deletion_generation'"
                ),
                {"schema": settings.DB.SCHEMA},
            ).all()
        )


def test_upgrade_builds_valid_indexes_and_downgrade_removes_them(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    assert _index_states(alembic_engine) == {}
    assert _generation_tables(alembic_engine) == set()

    command.upgrade(alembic_cfg, REVISION)

    states = _index_states(alembic_engine)
    assert set(states) == set(INDEXES)
    assert all(valid for _, valid in states.values())
    assert _generation_tables(alembic_engine) == set(GENERATION_TABLES)

    command.downgrade(alembic_cfg, DOWN_REVISION)

    assert _index_states(alembic_engine) == {}
    assert _generation_tables(alembic_engine) == set()


@pytest.mark.parametrize("caller_owns_transaction", [False, True])
def test_invalid_leftover_is_rebuilt(
    alembic_cfg: Config, alembic_engine: Engine, caller_owns_transaction: bool
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    index = INDEXES[0]
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE INDEX {index} ON "{settings.DB.SCHEMA}".queue'
                + " USING gin (workspace_name, (payload -> 'observers'))"
            )
        )
        # What a CREATE INDEX CONCURRENTLY that failed part-way leaves behind.
        conn.execute(
            text(
                "UPDATE pg_index SET indisvalid = false WHERE indexrelid ="
                + f" '\"{settings.DB.SCHEMA}\".{index}'::regclass"
            )
        )
    invalid = _index_states(alembic_engine)[index]
    assert invalid[1] is False

    if caller_owns_transaction:
        with alembic_engine.begin() as conn:
            alembic_cfg.attributes["connection"] = conn
            command.upgrade(alembic_cfg, REVISION)
    else:
        command.upgrade(alembic_cfg, REVISION)

    rebuilt = _index_states(alembic_engine)[index]
    assert rebuilt[1] is True
    assert rebuilt[0] != invalid[0]
