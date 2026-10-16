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


def _partition_messages(alembic_engine: Engine) -> None:
    """Replace messages with a hash-partitioned table, as the multi-tenant schema has."""
    schema = settings.DB.SCHEMA
    with alembic_engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{schema}".messages CASCADE'))
        conn.execute(
            text(
                f'CREATE TABLE "{schema}".messages'
                + " (tenant_id text NOT NULL, id bigint NOT NULL, content text,"
                + " PRIMARY KEY (tenant_id, id)) PARTITION BY HASH (tenant_id)"
            )
        )
        for remainder in range(2):
            conn.execute(
                text(
                    f'CREATE TABLE "{schema}".messages_p{remainder}'
                    + f' PARTITION OF "{schema}".messages'
                    + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                )
            )


def _partition_index_validity(alembic_engine: Engine) -> list[bool]:
    """indisvalid of every partition's index attached to the parent index."""
    with alembic_engine.connect() as conn:
        return list(
            conn.scalars(
                text(
                    "SELECT i.indisvalid FROM pg_inherits h"
                    + " JOIN pg_index i ON i.indexrelid = h.inhrelid"
                    + " WHERE h.inhparent = to_regclass(:parent)"
                ),
                {"parent": f'"{settings.DB.SCHEMA}".{INDEX}'},
            )
        )


def test_partitioned_messages_builds_and_drops_without_concurrently(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    """CONCURRENTLY is refused on a partitioned table, so both directions run
    in the migration's transaction and cascade to every partition."""
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition_messages(alembic_engine)

    command.upgrade(alembic_cfg, REVISION)

    state = _index_state(alembic_engine)
    assert state is not None and state[1] is True
    assert _partition_index_validity(alembic_engine) == [True, True]

    command.downgrade(alembic_cfg, DOWN_REVISION)

    assert _index_state(alembic_engine) is None


def test_partitioned_invalid_parent_from_unfinished_attach_is_rebuilt(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    """The non-blocking prebuild (ON ONLY + per-partition + ATTACH) leaves the
    parent INVALID until every partition is attached; if it stops part-way the
    revision must rebuild rather than skip it as already existing."""
    schema = settings.DB.SCHEMA
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition_messages(alembic_engine)
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE INDEX {INDEX} ON ONLY "{schema}".messages'
                + " USING gin (content gin_trgm_ops)"
            )
        )
        conn.execute(
            text(
                f'CREATE INDEX messages_p0_trgm ON "{schema}".messages_p0'
                + " USING gin (content gin_trgm_ops)"
            )
        )
        conn.execute(
            text(
                f'ALTER INDEX "{schema}".{INDEX} ATTACH PARTITION "{schema}".messages_p0_trgm'
            )
        )
    invalid = _index_state(alembic_engine)
    assert invalid is not None and invalid[1] is False

    command.upgrade(alembic_cfg, REVISION)

    rebuilt = _index_state(alembic_engine)
    assert rebuilt is not None and rebuilt[1] is True
    assert _partition_index_validity(alembic_engine) == [True, True]
