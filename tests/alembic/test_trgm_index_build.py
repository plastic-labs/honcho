"""The concurrent build path of revision 3e7a1c9d5b20 (messages.content trigram index).

test_pipeline.py runs revisions on a caller-owned connection, which takes the
transactional CREATE INDEX branch. These run ``command.upgrade`` the way
``src.migrate`` does, so env.py owns the transaction and the revision builds
the index CONCURRENTLY in an autocommit block.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings

REVISION = "3e7a1c9d5b20"
DOWN_REVISION = "b8d2f4a6c9e1"
INDEX = "ix_messages_content_trgm"


def _index_state(alembic_engine: Engine) -> tuple[int, bool] | None:
    """Return the index's (oid, indisvalid), or None if it doesn't exist."""
    with alembic_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT i.indexrelid::bigint, i.indisvalid FROM pg_index i"
                + " JOIN pg_class c ON c.oid = i.indexrelid"
                + " JOIN pg_namespace n ON n.oid = c.relnamespace"
                + " WHERE c.relname = :name AND n.nspname = :schema"
            ),
            {"name": INDEX, "schema": settings.DB.SCHEMA},
        ).first()
    return None if row is None else (row[0], row[1])


def _create_index(alembic_engine: Engine) -> None:
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE INDEX {INDEX} ON "{settings.DB.SCHEMA}".messages'
                + " USING gin (content gin_trgm_ops)"
            )
        )


def test_concurrent_build_creates_valid_index(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    assert _index_state(alembic_engine) is None

    command.upgrade(alembic_cfg, REVISION)

    state = _index_state(alembic_engine)
    assert state is not None and state[1] is True


def test_invalid_leftover_is_rebuilt(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _create_index(alembic_engine)
    # What a CREATE INDEX CONCURRENTLY that failed part-way leaves behind.
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE pg_index SET indisvalid = false WHERE indexrelid ="
                + f" '\"{settings.DB.SCHEMA}\".{INDEX}'::regclass"
            )
        )
    invalid = _index_state(alembic_engine)
    assert invalid is not None and invalid[1] is False

    command.upgrade(alembic_cfg, REVISION)

    rebuilt = _index_state(alembic_engine)
    assert rebuilt is not None and rebuilt[1] is True
    assert rebuilt[0] != invalid[0]


def test_prebuilt_index_is_kept(alembic_cfg: Config, alembic_engine: Engine) -> None:
    """An index built ahead of the release makes the revision a no-op."""
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _create_index(alembic_engine)
    prebuilt = _index_state(alembic_engine)
    assert prebuilt is not None

    command.upgrade(alembic_cfg, REVISION)

    assert _index_state(alembic_engine) == prebuilt
