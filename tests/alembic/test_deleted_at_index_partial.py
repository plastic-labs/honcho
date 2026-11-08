"""Revision a3f9c2e81d47 makes a full ix_documents_deleted_at partial.

A database built from the models (not by the migrations) had a full index on
documents.deleted_at. The revision must replace it with the partial index that
revision 119a52b73c60 makes.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings

REVISION = "a3f9c2e81d47"
DOWN_REVISION = "647b773f05ac"
INDEX = "ix_documents_deleted_at"


def _is_partial(alembic_engine: Engine) -> bool | None:
    """True if the index has a WHERE clause, None if it does not exist."""
    with alembic_engine.connect() as conn:
        return conn.scalar(
            text(
                "SELECT indpred IS NOT NULL FROM pg_index"
                + " WHERE indexrelid = to_regclass(:name)"
            ),
            {"name": f'"{settings.DB.SCHEMA}".{INDEX}'},
        )


def test_full_deleted_at_index_becomes_partial(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    schema = settings.DB.SCHEMA
    command.upgrade(alembic_cfg, DOWN_REVISION)
    with alembic_engine.begin() as conn:
        conn.execute(text(f'DROP INDEX "{schema}".{INDEX}'))
        conn.execute(text(f'CREATE INDEX {INDEX} ON "{schema}".documents (deleted_at)'))
    assert _is_partial(alembic_engine) is False

    command.upgrade(alembic_cfg, REVISION)

    assert _is_partial(alembic_engine) is True


def test_partial_deleted_at_index_is_kept(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    assert _is_partial(alembic_engine) is True
    with alembic_engine.connect() as conn:
        before = conn.scalar(
            text("SELECT to_regclass(:name)::oid::bigint"),
            {"name": f'"{settings.DB.SCHEMA}".{INDEX}'},
        )

    command.upgrade(alembic_cfg, REVISION)

    with alembic_engine.connect() as conn:
        after = conn.scalar(
            text("SELECT to_regclass(:name)::oid::bigint"),
            {"name": f'"{settings.DB.SCHEMA}".{INDEX}'},
        )
    assert after == before


def test_conversion_that_stopped_before_the_rename_is_finished(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    """A run can stop after it drops the full index and before the rename."""
    schema = settings.DB.SCHEMA
    command.upgrade(alembic_cfg, DOWN_REVISION)
    with alembic_engine.begin() as conn:
        conn.execute(text(f'DROP INDEX "{schema}".{INDEX}'))
        conn.execute(
            text(
                f'CREATE INDEX {INDEX}_partial ON "{schema}".documents (deleted_at)'
                + " WHERE deleted_at IS NOT NULL"
            )
        )
    assert _is_partial(alembic_engine) is None

    command.upgrade(alembic_cfg, REVISION)

    assert _is_partial(alembic_engine) is True
    with alembic_engine.connect() as conn:
        leftover = conn.scalar(
            text("SELECT to_regclass(:name)"),
            {"name": f'"{schema}".{INDEX}_partial'},
        )
    assert leftover is None


def _partition_documents(alembic_engine: Engine) -> None:
    """Replace documents with a hash-partitioned copy, as the multi-tenant schema has."""
    schema = settings.DB.SCHEMA
    with alembic_engine.begin() as conn:
        conn.execute(
            text(f'ALTER TABLE "{schema}".documents RENAME TO documents_plain')
        )
        conn.execute(
            text(
                f'CREATE TABLE "{schema}".documents'
                + f' (LIKE "{schema}".documents_plain INCLUDING DEFAULTS)'
                + " PARTITION BY HASH (tenant_id)"
            )
        )
        conn.execute(
            text(f'ALTER TABLE "{schema}".documents ADD PRIMARY KEY (tenant_id, id)')
        )
        for remainder in range(2):
            conn.execute(
                text(
                    f'CREATE TABLE "{schema}".documents_p{remainder}'
                    + f' PARTITION OF "{schema}".documents'
                    + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                )
            )
        conn.execute(text(f'DROP TABLE "{schema}".documents_plain CASCADE'))
        conn.execute(text(f'CREATE INDEX {INDEX} ON "{schema}".documents (deleted_at)'))


def test_partitioned_conversion_runs_again_after_the_rename(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    """After the rename, the partition indexes keep the temporary name. A second
    run (a retry after a later step failed, or downgrade then upgrade) must not
    try to drop them."""
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition_documents(alembic_engine)
    assert _is_partial(alembic_engine) is False

    command.upgrade(alembic_cfg, REVISION)
    assert _is_partial(alembic_engine) is True

    # A retry: the revision ran, but it is not recorded.
    command.stamp(alembic_cfg, DOWN_REVISION)
    command.upgrade(alembic_cfg, REVISION)
    assert _is_partial(alembic_engine) is True

    command.downgrade(alembic_cfg, DOWN_REVISION)
    command.upgrade(alembic_cfg, REVISION)
    assert _is_partial(alembic_engine) is True
